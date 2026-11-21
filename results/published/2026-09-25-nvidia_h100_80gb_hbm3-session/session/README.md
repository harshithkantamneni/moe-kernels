# One of the study's four cards, the second Hopper: DRAM counters and clock-locked R3 on one NVIDIA H100 80GB HBM3 (Lambda Cloud); every measurement here is this card's, never averaged with another card's

Every measured number in this directory is the H100 80GB HBM3's. One number
here is not this card's: ALPHA_BAND [0.529, 0.588), the study's refit band for
alpha around 0.558 (`scripts/block_m_crossing_sweep.py` at f49a213), fitted on
2026-09-01 over 10,813 rows of the study's published pages (`docs/FINDINGS.md`,
"REFIT 2026-09-01"). Each timed R3 page's C1 gate scores this card's ratio
against it, and it appears here only where C1 does. Apart from that band, no
number from the study's H200 pages is set beside these.

Lambda Cloud instance b10f934c7fb84a959287768e8bf1563f, named moe-h100, type
gpu_1x_h100_sxm5 (1x H100 80 GB SXM5), region us-south-2 (North Texas), $4.29/h,
26 vCPUs, 225 GiB RAM. It was launched at 13:35:20 UTC on 2026-09-25 by a laptop
capacity watch, under the owner's explicit permission for one H100, was active
at 13:38:09 UTC, and had the scripts copied to it before the autopilot started;
the exfil was verified and termination requested at 15:09:50 UTC (about 1.58 h;
at $4.29/h, about $6.8). The instance id, name, type, region, price, vCPU and
RAM figures, the launch by a laptop capacity watch under the owner's explicit
permission for one H100, the launch and active times, the copy of the scripts,
and the exfil check and termination request at 15:09:50 UTC are from the
owner's session; no file here records them. `drivers/h100_autopilot.sh`
carries the same 4.29 as its `RATE` default, and `setup_vm.log` prints the host
as `x86_64` and `221 GB RAM`.

The card, from `PREFLIGHT.json`: NVIDIA H100 80GB HBM3,
GPU-4e782590-2a78-3dae-2f1b-a3ce1000a28f, capability 9.0 (sm_90), 132 SMs,
50 MiB L2 (52428800 bytes). From
`alpha_g-nvidia_h100_80gb_hbm3-20260925T134737Z/logs/calibrate.log`: 700 W power
limit, 1980 MHz maximum SM clock, 2619 MHz memory clock.

Host (`setup_vm.log`): Ubuntu 22.04, Lambda stack 0.1.17 (its system torch 2.7.0
is recorded and never used). Venvs torch 2.13.0+cu130 and triton 3.7.1 (base),
plus vLLM 0.27.1 (vllm), Python 3.12.14, CUDA 13.0. Nsight Compute 2025.3.1.0
(build 36398880) at `/usr/local/cuda-13.0/bin/ncu`, installed by `setup_vm.sh`
S5 as `cuda-nsight-compute-13-0`. Counter door `sudo`: uid 1000 had CapEff 0
(`setup_vm.log` S0 and S6).

Driver: the autopilot found driver 570 and installed
`nvidia-driver-580-server-open` 580.105.08 from Lambda's repository, then
rebooted (`autopilot.log`; the apt output is `driver580.log`, which also lists
`nvidia-firmware-570-server-570.148.08` as "automatically installed and is no
longer required"). The full old version, 570.148.08, and the `@reboot` crontab
line that restarted the autopilot after the reboot are from the owner's session;
`drivers/h100_autopilot.sh` says in its header that such a line restarts it, and
`autopilot.log` shows the restart. Every S0 block in `setup_vm.log` reads
`580.105.08 (CUDA 13.0 per nvidia-smi)`. The package and repository are the
ones the A100 counter run was upgraded with
(`../../2026-09-25-nvidia_a100_sxm4_40gb-r3-counters/session/README.md`).

Code: `setup_vm.sh` S1 fetched the git bundle `/home/ubuntu/moe3.bundle` and
checked out f49a213 detached and clean (`setup_vm.log`). Every page here names
f49a213.

## What is here

Paths inside the logs are the VM's: `/home/ubuntu/moe/session/<x>` is `./<x>`
here, `/home/ubuntu/moe/results/<x>` is `../results/<x>`,
`/home/ubuntu/moe/repo/results/calibration/<x>` is `../calibration/<x>`,
`/home/ubuntu/moe/alpha_g_chain.out` and the files in `/home/ubuntu` are `./`,
and `/home/ubuntu/moe/repo` was the checkout.

- `../KIND`: `session`. See "Why KIND is session" below.
- `../results/2026-09-25-nvidia_h100_80gb_hbm3-r3-counters/`: the run directory
  `run_counters.sh` wrote. `r3c-g{1,2,4,16,64}.json` is one page per G;
  `r3c-g<G>.profiles/` holds that G's `g<G>.ncu-rep`, the raw `g<G>.csv`,
  `g<G>.capture.json` (ncu's argv, version and return code), `g<G>.plan.json`,
  `g<G>.manifest.json`, `g<G>.ncu.log` and the Triton cache; `summary.json` is
  the `--analyse` output over the five pages.
- `../results/gaps-nvidia_h100_80gb_hbm3/`: the timed arms' run directories.
  - `thermal_acceptance/...-91e83e4a/`: thermal's report.
  - `private_weight_reference/`: 10 R3 run directories, one per attempt: the
    chain's `1505594b` (step 4 of "What ran, in order") and the nine
    `locked_r3.py` attempts (the table in step 5). 7 are complete pages with
    `report.json` (the table under "The readings"); 3 were stopped:
    `bf2b7e51` (2 cells), `f928846e` (10 cells), and `e0a0087e`, which holds
    only `DEVICE` and an empty `triton-cache/` that git does not keep.
  - `clock_elasticity/...-de1fff67/`: `CARD` and `DEVICE` only. R1 at G=1
    created it at 14:02:47 and was stopped before it measured anything.
- `../calibration/`: the ruler the chain's calibrate measured and its run
  directory; NOT adopted, see its README.
- The VM's session root, `/home/ubuntu/moe/session`, whole:
  - `PREFLIGHT.txt`, `PREFLIGHT.json`: the preflight at 13:43:49Z, READY.
  - `probe.json` and `logs/pf5-probe.log`: PF5's route probe.
  - `census.json`, `census.profiles/` and `logs/pf6-census.log`: PF6's launch
    census.
  - `ncu.txt` (the ncu binary, its version and `--list-chips`),
    `ncu-query-metrics.txt`.
  - `logs/setup_runpod.log` (the venv build `setup_vm.sh` S4 ran),
    `logs/r3c-g<G>.log` (one per G), `logs/analyse.log`.
  - `alpha_g-nvidia_h100_80gb_hbm3-20260925T134737Z/`: the chain session:
    `CHAIN.tsv` (its ledger), `ARMS.tsv` (the preconditions driver's), `DEVICE`,
    `R3_DUTY` (0.25), `COUNTERS` and `COUNTERS.json` (the chain's counter
    probe), `chain.lock` (`9396 192-222-53-184`: a PID and the VM's hostname),
    `chain-logs/` (one log per step, plus each arm's `.price.log`; `r1-g1.log`
    is 0 bytes), `logs/` (thermal, calibrate, pin_probe-n64-g1 and
    `counter_plan.log`) and `pin-probe-n64-g1/`.
  - `locked-r3/`: what `locked_r3.py` wrote. `status` (one line per step),
    `summary.json` (every attempt and the kept pages), `supported-clocks.txt`,
    `lgc-<L>.txt` and `clocks-locked-<L>.txt` for each lock, `rgc.txt` and
    `clocks-after-reset.txt`, `r3-g<G>-lock<L>.log` for each attempt, and
    `smi.csv`, nvidia-smi sampled every 500 ms (SM and memory clock, power,
    temperature, utilisation, clock event reasons) from 14:03:07.886 to
    15:08:55.808, 7888 samples.
- The consoles:
  - `autopilot.log` (the autopilot's stages), `autopilot.state` (`done`),
    `autopilot.out` (0 bytes), `driver580.log`, `setup_vm.log`,
    `run_counters.status` (the counter run's ledger), `run_counters.out` (0
    bytes), `locked_r3.out` (0 bytes): the tarball's top level, `/home/ubuntu`
    on the VM.
  - `alpha_g_chain.out`: the chain's console, from `/home/ubuntu/moe`.
- `drivers/`: `h100_autopilot.sh` and `locked_r3.py`. See "The driver scripts".
- `EXFIL-NOT-PUBLISHED.txt`: every tarball file not published here, with the
  reason.

## What ran, in order

Times are UTC on 2026-09-25, from `autopilot.log`, `setup_vm.log`,
`run_counters.status`, `locked-r3/status`, the pages' `provenance.utc`, and,
for the chain's steps, `CHAIN.tsv` and `ARMS.tsv` read with the file mtimes the
tarball carries.

1. **Driver** (`autopilot.log`). 13:38:25 autopilot start at stage `driver`,
   "driver 570: installing nvidia-driver-580-server-open from Lambda's repo";
   13:40:01 stage -> setup, "rebooting for the new driver"; 13:41:19 autopilot
   start at stage `setup`.
2. **`setup_vm.sh` at f49a213** (`setup_vm.log`), 13:41:19 to 13:43:49 by
   `autopilot.log` (2 min 30 s). S1 checked the bundle out; S4 built the base
   and vllm venvs from `requirements/resolved-*.txt`; S5 installed ncu; S6 chose
   the `sudo` counter door; S7 wrote `env.sh`. S8, the preflight, at
   13:43:49Z: PF1 to PF7 PASS, verdict READY (exit 0). PF7: "the card keeps 59.2
   GB free beside the 25.8 GB footprint, so the save stays on the device".
   The log ends `EXIT 0`.
3. **Bytes: `run_counters.sh`** (`run_counters.status`).
   `scripts/dram_counter_route.py --run --family r3-arms --group-m G --census
   /home/ubuntu/moe/session/census.json` under the `sudo` door, one G at a time
   in the order 64, 1, 4, 2, 16, then `--analyse` over the five pages:

   | G | start | end | wall | exit |
   |---|---|---|---|---|
   | 64 | 13:43:49 | 13:44:34 | 45 s | 3 |
   | 1 | 13:44:34 | 13:45:19 | 45 s | 3 |
   | 4 | 13:45:19 | 13:46:04 | 45 s | 3 |
   | 2 | 13:46:04 | 13:46:49 | 45 s | 3 |
   | 16 | 13:46:49 | 13:47:34 | 45 s | 3 |
   | `--analyse` | | 13:47:36 | | 3 |

   Every page names commit f49a213 with `git_dirty: false`, every
   `capture.json` names f49a213 with ncu return code 0 and no metric dropped,
   and no `g<G>.ncu.log` nor `census.ncu.log` carries a `WARNING` line.
4. **The alpha(G) chain at G=1, seed 0** (`alpha_g_chain.out`, `CHAIN.tsv`,
   `ARMS.tsv`). 13:47:36 stage -> pre. The autopilot started
   `scripts/alpha_g_chain.sh` with `G_LADDER=1 SEEDS=0` and `RATE_USD_H` set
   from its `RATE`, default 4.29 (`drivers/h100_autopilot.sh`); the console
   prints session
   `alpha_g-nvidia_h100_80gb_hbm3-20260925T134737Z`, "G in {1}, seeds {0}, ratio
   arms at duty 0.25", and nvidia-smi as "NVIDIA H100 80GB HBM3, 580.105.08,
   700.00 W".
   - preflight-r1 DONE (2 s), preflight-r3 DONE (5 s).
   - preconditions DONE (196 s, log closed 13:51:02). The work tree had 0 dirty
     files before anything ran (`chain-logs/preconditions.log`).
     - thermal DONE (153 s, log closed 13:50:22): 1380 MHz median (min 1335,
       max 1395) over 55 samples, first and last third 1380 MHz, 0.0% apart,
       board power median 699 W of a 700 W limit, 69 to 72 C (`logs/thermal.log`).
     - calibrate DONE (30 s; the yaml's `provenance.utc` is 13:50:52):
       `clock_established PASS`, bf16 GEMM clock samples 1395, 1365, 1380,
       1365, 1380 MHz (median 1380, spread 2.1%), triad 3101.5 GB/s, bf16 694.4
       TFLOP/s, ridge 223.9 (`logs/calibrate.log`). It printed `PUBLISHED to
       /home/ubuntu/moe/repo/moe/bench/hardware/measured_nvidia_h100_80gb_hbm3.yaml`,
       and the work tree went from 0 to 1 dirty file. See `../calibration/README.md`.
     - pin_probe-n64-g1 DONE (9 s): F1 and F2 PASS.
   - counter-probe INFO (rc 1, 5 s, 13:51:07): BLOCKED, `ERR_NVGPUCTRPERM`,
     CapEff 0000000000000000 (`COUNTERS`). The chain runs its probe as uid 1000
     without the `sudo` door the counter run used; the probe is informational
     and gates nothing.
   - gpu-tests DONE (49 s): "37 passed, 1 skipped, 2 warnings in 45.83s".
   - probe-check DONE (8 s): P1 PASS, 5.20 us per call under the graph.
   - r3-g1-s0 CLAIM_FAIL (rc 1, 634 s): page `1505594b`, 13:52:10 to 14:02:38.
   - r1-g1: its price step wrote `chain-logs/r1-g1.price.log` (a `--dry-run`:
     "Nothing was measured and nothing was written"), and R1 created
     `clock_elasticity/...-de1fff67` at 14:02:47. `chain-logs/r1-g1.log` is 0
     bytes and `CHAIN.tsv` has no r1-g1 row.
   - `autopilot.log`: 14:02:47 "pre: r3-g1-s0 row written in
     /home/ubuntu/moe/session/alpha_g-nvidia_h100_80gb_hbm3-20260925T134737Z/;
     stopping the chain before R1", 14:03:07 "pre: chain session 9394 stopped",
     stage -> locked.
5. **Locked R3: `locked_r3.py`** (`locked-r3/status`, `locked-r3/summary.json`).
   14:03:07 "lock ladder [1980, 1890, 1800, 1710, 1620, 1530] (supported max
   1980)". For each lock L it ran `sudo -n nvidia-smi -lgc L,L` (each
   `lgc-<L>.txt` reads `GPU clocks set to "(gpuClkMin L, gpuClkMax L)"`), then R3
   at G = 1, 2, 4, 16, 64 in turn: `scripts/private_weight_reference.py --model
   mixtral-8x7b --block-m 32 --treads 6 --repeats 9 --group-m G --duty 0.25
   --seed 0 --session-tag <chain session>-lock<L>`. Every 10 s it read the run's
   `cells.csv`; a timed cell whose `sm_clock_load_mhz` is more than one 15 MHz
   step under L stopped the run, and the ladder moved to the next lower lock and
   began again at G=1 (`drivers/locked_r3.py`).

   | lock | G | `status` line (start to end) | run directory | cells and their clocks (`cells.csv`) |
   |---|---|---|---|---|
   | 1980 | 1 | 14:03:08 to 14:03:39, "slipped rc=None cells=2 worst=1830.0 32 s" | `bf2b7e51` | 2 cells, both 1830 MHz |
   | 1890 | 1 | 14:03:39 to 14:03:51, "slipped rc=None cells=2 worst=1830.0 12 s" | `e0a0087e` (`summary.json` names `bf2b7e51`) | none: no `cells.csv` |
   | 1800 | 1 | 14:03:51 to 14:14:32, "done rc=1 cells=162 worst=1800.0 641 s" | `373afb62` | 162 cells, all 1800 MHz |
   | 1800 | 2 | 14:14:32 to 14:15:32, "slipped rc=None cells=10 worst=1770.0 60 s" | `f928846e` | 10 cells: 9 at 1800, 1 at 1770 MHz |
   | 1710 | 1 | 14:15:32 to 14:26:13, "done rc=1 cells=162 worst=1710.0 641 s" | `ff84f0aa` | 162 cells, all 1710 MHz |
   | 1710 | 2 | 14:26:13 to 14:36:53, "done rc=0 cells=162 worst=1710.0 641 s" | `0708f122` | 162 cells, all 1710 MHz |
   | 1710 | 4 | 14:36:53 to 14:47:34, "done rc=0 cells=162 worst=1710.0 641 s" | `48bc690e` | 162 cells, all 1710 MHz |
   | 1710 | 16 | 14:47:34 to 14:58:15, "done rc=0 cells=162 worst=1710.0 641 s" | `22da27e7` | 162 cells, all 1710 MHz |
   | 1710 | 64 | 14:58:15 to 15:08:56, "done rc=1 cells=162 worst=1710.0 641 s" | `ceb347ad` | 162 cells, all 1710 MHz |

   15:08:56 "all G done at lock 1710 MHz", "clock reset; DONE rc=0". 1620 and
   1530 were on the ladder and never locked. `rgc.txt` reads "All done."; the
   query `clocks-after-reset.txt` took in the same second still reads SM 1710
   MHz, and no later clock reading is in the tarball. The autopilot then ran
   `sudo nvidia-smi -rgc` twice more (`drivers/h100_autopilot.sh` lines 83 and
   88, before it logged "locked_r3.py rc=0" and "ALL DONE"), with its output
   discarded, so no file records what those two resets read.
6. **Done** (`autopilot.log`). 15:08:56 "locked_r3.py rc=0", stage -> done, "ALL
   DONE: results wait on disk for the laptop to copy off". `autopilot.state`
   reads `done`.

### The 1890 attempt measured nothing

No 1890 MHz result here is a measurement. `locked_r3.py`'s `run_dir` (lines 73
to 80 of `drivers/locked_r3.py`) picks the newest `*-duty0.25-g<G>-*` directory
whose `cells.csv` was modified at or after the attempt's start less 5 s, and it
does not check the session tag. The 1980 attempt's last cell landed in
`bf2b7e51/cells.csv` at 14:03:35 (the tarball's mtime) and the 1890 attempt
started at 14:03:39, inside those 5 s. So the 1890 attempt's first check read
the 1980 attempt's 2 cells at 1830 MHz, called them a slip and stopped the run
12 s in: `summary.json` names `bf2b7e51` as the 1890 attempt's directory, and
its status line repeats the 1980 attempt's "cells=2 worst=1830.0". The 1890
run's own directory, `e0a0087e`, holds only `DEVICE` (14:03:43) and an empty
`triton-cache/`, and `r3-g1-lock1890.log` ends in a `KeyboardInterrupt` inside
`probe_cells`, before its first timed cell. `smi.csv`'s 24 samples in that
window read 1830 MHz; they are nvidia-smi's, not a cell's.

### What the clocks read beside the cells

The lock rule scores each timed cell's `sm_clock_load_mhz`. `smi.csv`, sampled
beside it, adds:

- Under the 1980 lock the card never read 1980 MHz: all 62 `smi.csv` samples
  from 14:03:08 to 14:03:38 read 1830 MHz, at no more than 317 W against the
  700 W limit. `clocks-locked-1980.txt`, taken at 14:03:08 right after the
  lock, also reads SM 1830 MHz with the card idle (`Idle: Active`, 72.38 W
  average power draw). The same query right after the 1890 and 1800 locks
  (`clocks-locked-1890.txt`, `clocks-locked-1800.txt`) reads 1830 MHz too,
  while `smi.csv` reads 1800 MHz from 14:03:51.449, so that query alone does
  not show whether a lock took.
- Under the 1800 lock at G=2, 5 of the 120 samples read under 1800 MHz (1740
  to 1785 MHz).
- Under the 1710 lock, from 14:15:33 to 15:08:56, 65 of 6399 samples read under
  1710 MHz, 50 of them under 1695 MHz; the lowest is 1545 MHz, during G=2.
  122 samples in that window carry clock event reason `0x0000000000000004`
  (SwPowerCap, as the chain's
  `alpha_g-nvidia_h100_80gb_hbm3-20260925T134737Z/logs/thermal.log` names
  0x4), while board power peaks at 385 W. Every timed cell of the five 1710
  pages reads 1710 MHz.

## The readings

### R3, timed (`private_weight_reference`)

mixtral-8x7b, bf16, BLOCK_M=32, 6 treads x 3 arms x 9 repeats = 162 cells,
duty 0.25, seed 0. The ratio is slope(shared) / slope(private) over treads 2 to
6, with its 90% percentile bootstrap interval; C1's band is ALPHA_BAND [0.529,
0.588). Each page is one run scored alone: its interval is over repeats within
the run, not a run-to-run spread. Verdicts, ratios and rates are as each
`report.txt` prints them; the times are each `report.json`'s `provenance.utc`
to its mtime.

| G | lock | directory | UTC | exit as printed | ratio [90%] | C1 world | C2: private arm's delivered rate |
|---|---|---|---|---|---|---|---|
| 1 | none | `1505594b` | 13:52:10 to 14:02:38 | 1 CLAIM_FAIL | 0.9516 [0.9507, 0.9529] | NO-REUSE | 2955.4 GB/s |
| 1 | 1800 | `373afb62` | 14:03:54 to 14:14:22 | 1 CLAIM_FAIL | 0.9519 [0.9513, 0.9537] | NO-REUSE | 2956.4 GB/s |
| 1 | 1710 | `ff84f0aa` | 14:15:35 to 14:26:04 | 1 CLAIM_FAIL | 0.9508 [0.9500, 0.9511] | NO-REUSE | 2954.5 GB/s |
| 2 | 1710 | `0708f122` | 14:26:16 to 14:36:46 | 0 DONE | 0.5773 [0.5772, 0.5774] | REFIT-CONFIRMED | 2942.9 GB/s |
| 4 | 1710 | `48bc690e` | 14:36:57 to 14:47:30 | 0 DONE | 0.5657 [0.5654, 0.5658] | REFIT-CONFIRMED | 2920.7 GB/s |
| 16 | 1710 | `22da27e7` | 14:47:38 to 14:58:13 | 0 DONE | 0.5407 [0.5405, 0.5408] | REFIT-CONFIRMED | 2800.7 GB/s |
| 64 | 1710 | `ceb347ad` | 14:58:18 to 15:08:52 | 1 CLAIM_FAIL | 0.4827 [0.4814, 0.4830] | BELOW-THE-REFIT-BAND | 2521.0 GB/s |

On all seven pages V0 to V8 PASS, V7 (shared and private under-load clocks
within 1% at every tread) measures worst 0.00%, and V6 (shared and private at
n=1) measures a 0.006% to 0.013% gap. C2 PASSES on every page, against a gate
of 3411.7 GB/s (10% over the calibrated 3101.5). A CLAIM_FAIL page reads "the
gates imply 1 CLAIM_FAIL: measured; VALIDITY passed; a CLAIM gate did not (a
result, not a retry)", and C1 is the gate that failed on each; a DONE page
reads "the gates imply 0 DONE: measured; every VALIDITY and CLAIM gate PASSED".

`1505594b` is the chain's page, unlocked: all 162 of its cells read 1980 MHz.
It is kept as this card's record of duty 0.25 without a lock beside the same
geometry under one. `373afb62` is a complete G=1 page at 1800 MHz; the 1800
ladder stopped at G=2, so it is the only 1800 MHz page.

### R3 under DRAM counters (`dram_counter_route.py`)

The alpha(G) table as `logs/analyse.log` prints it (lines 1 to 10). These alphas
are THIS card's, measured by DRAM counters on the H100 80GB HBM3; they are not
the study's H200 alphas.

```
CARD NVIDIA H100 80GB HBM3 (nvidia_h100_80gb_hbm3, UUID 4e782590-2a78-3dae-2f1b-a3ce1000a28f, sm_90, 132 SMs, 50 MiB L2): every number here is THIS card's; the study's timing pages are nvidia_h200.
ALPHA(G) OVER 5 PAGES, one card, one commit, one vLLM, one design
  alpha lo/hi: the least and greatest OLS slope over SHARED's weight-only q, lowest call of SHARED or NATIVE less e to the highest,
  e PRIVATE's excess over n at its highest call; slope: q_S's K-call means', every activation re-read counted as a weight re-read
     G  alpha lo  alpha hi    slope       w1       w2    resid    ratio     diff  exit
     1    0.8293    0.8364   0.8327   0.9915   0.5152    4.62%   0.8321   0.8313  3 INVALID: measured; a VALIDITY gate failed after measuring; nothing quotable
     2    0.4046    0.4118   0.4101   0.4197   0.3911   27.82%   0.4107   0.4067  3 INVALID: measured; a VALIDITY gate failed after measuring; nothing quotable
     4    0.1740    0.1887   0.1866   0.1838   0.1923   43.55%   0.1876   0.1773  3 INVALID: measured; a VALIDITY gate failed after measuring; nothing quotable
    16    0.0086    0.0735   0.0673   0.0440   0.1139   10.62%   0.0670   0.0151  3 INVALID: measured; a VALIDITY gate failed after measuring; nothing quotable
    64   -0.0872    0.0831   0.0716   0.0013   0.2122    2.74%   0.0653  -0.0725  3 INVALID: measured; a VALIDITY gate failed after measuring; nothing quotable
```

All five pages are INVALID, so `--analyse` exited 3. The gates, as
`logs/analyse.log` prints them, scored by the counter scorer at f49a213. V0 to
V5, V8 and V9 PASS on every page; V7 PASSES on G = 1, 2, 4 and 16. V6's gate is
"max/min - 1 <= 0.5% across arms".

| G | exit | VALIDITY gates that FAIL | CLAIM gates |
|---|---|---|---|
| 1 | 3 INVALID | V6: n=2 w1 0.6375%; n=2 w2 3.1191%; n=3 w1 0.9492%; n=3 w2 2.5081%; and 4 more | C2 PASS, C3 PASS, C6 PASS |
| 2 | 3 INVALID | V6: n=2 w1 7.3262%; n=2 w2 10.5400%; n=3 w1 0.6049%; n=3 w2 8.9846%; and 4 more | C1 PASS, C2 PASS, C6 PASS |
| 4 | 3 INVALID | V6: n=2 w1 7.4466%; n=2 w2 9.3683%; n=3 w1 6.3973%; n=3 w2 6.4568%; and 4 more | C1 PASS, C2 UNKNOWN (its GATE line prints REFUSE), C6 PASS |
| 16 | 3 INVALID | V6: n=2 w1 6.9725%; n=2 w2 11.3474%; n=3 w1 4.4440%; n=3 w2 4.9955%; and 4 more | C1 UNKNOWN (REFUSE), C2 UNKNOWN (REFUSE), C6 FAIL |
| 64 | 3 INVALID | V6: n=1 w1 1.3753%; n=1 w2 0.7424%; n=2 w1 5.8605%; n=2 w2 10.2500%; and 6 more. V7: n=2 w2 -7.5990% against 6.0409%, and more | C1 UNKNOWN (REFUSE), C2 FAIL, C6 FAIL |

A change to how V6 scores Hopper cards is under review (from the owner's
session; no file here records it). It is not applied here, and every verdict
above is the one the scorer at f49a213 printed on the VM.

The per-(G, arm, GEMM) q tables and each gate's measured values and reasons
follow the alpha table in `logs/analyse.log`; the same are in each
`logs/r3c-g<G>.log`. Each of those logs, and `logs/analyse.log`, ends with `git
check-ignore exited 128; path UNVERIFIED`: the results root
`/home/ubuntu/moe/results` is outside the checkout `/home/ubuntu/moe/repo`.

How each counter page was captured, from `g<G>.capture.json` and
`g<G>.plan.json`: ncu's argv includes `--replay-mode kernel --cache-control all
--clock-control base --nvtx -k regex:^fused_moe_kernel$ --launch-skip 60
--launch-count 90` over 16 metrics, the child being
`scripts/private_weight_reference.py --counter-child g<G>.plan.json` in the vllm
venv. mixtral-8x7b, bf16, BLOCK_M=32, BLOCK_N=64, num_stages=4; arms native,
shared and private at treads 1, 2, 3, 4 and 6 (15 cells); 3 calls per cell
after 2 warmups; 9 copies declared.

## Ruler and tree state

The five counter pages and `summary.json` carry `git_sha` f49a213 and
`git_dirty: false`, and record `ridge_source` and `bandwidth_source` as null:
they ran before calibrate and use no ruler. Every one of the 7 R3 `report.json`
files carries `git_sha` f49a213, `git_dirty: true` and `git_dirty_files: 1`
(calibrate's yaml, written into the checkout), and cites the chain's calibrate:
ridge 223.89, bandwidth 3101.5 GB/s, roof 694.39 TFLOP/s, reference clock 1380
MHz. That ruler is kept in `../calibration/` and is NOT adopted: nothing under
`moe/bench/hardware/` changes with this directory.

## The driver scripts

`drivers/h100_autopilot.sh` and `drivers/locked_r3.py` are the laptop's copies
of what was copied to the VM at launch; the tarball holds none of the scripts
the VM ran from `/home/ubuntu`. The autopilot also ran two more. They are not
duplicated here because the laptop's copy of each is byte-identical to a file
in the repository at f49a213 (`git hash-object` on the laptop's copy against
`git rev-parse f49a213:<path>`):

- `setup_vm.sh` is `scripts/setup_vm.sh` (blob f7de6b0dd93a3b7b2dd9a30d864c4e96189ee136).
- `run_counters.sh` is
  `results/published/2026-09-25-nvidia_a100_sxm4_40gb-r3-counters/session/run_counters.sh`,
  the A100 counter run's driver (blob 452461f94099176f5843e5902d11b9c664456859).
  The tarball's `moe/repo/` holds that A100 file with the same blob, listed in
  `EXFIL-NOT-PUBLISHED.txt`.

The laptop's own orchestration and logs are not published.

## Why KIND is session

The repository's KIND vocabulary has one word, `session`
(`moe/bench/published.py`, `SESSION` and `KIND_MARKER`). No word names this mix
of a counter run, a chain session and a lock search, and none is added here. The
docstring of `SESSION` describes a raw session directory that holds the
driver's ledger, its per-arm logs and the run directories the arms wrote, kept
so each verdict printed can be re-derived, with no `measured.yaml` and no
`merged.csv` of its own. This directory is that: `run_counters.status`,
`CHAIN.tsv`, `ARMS.tsv` and `locked-r3/status` are the ledgers, `logs/`,
`chain-logs/` and `locked-r3/` the logs, `../results/` the run directories, and
there is no `measured.yaml` or `merged.csv`.

With any other word, `calibration_provenance` reads the directory as an arm
without a calibration (verdict `unknown`, blocking), `tests/test_docs.py`
counts it as a published arm, and `scripts/rescore_published_reports.py` walks
its run directories' `report.json` files as published reports. With `session`,
the regenerated `results/published/CALIBRATION_PROVENANCE.md` lists it as
`session | n/a | n/a | n/a | **refused**` with the line "rows carry 0 dtypes [],
so there is no single ridge; pass one", and on this branch (r3-align at f49a213
plus this directory) its count reads 17 of 21.

## Provenance of these files

The exfil tarball `exfil.tgz`, written on the VM as `/home/ubuntu/exfil.tgz`,
sha256 `d13a011e563d45f74622956e2f3ba7336aa1a8a16d3b87581508c03c85e6ffad`, holds
6132 entries: 5021 files and 1111 directories. It was extracted fresh, and 406
of its files were copied here and compared by sha256 against that extraction:
406 identical, 0 different. The tarball maps as `moe/results/` to `../results/`,
`moe/session/` to `./`, `moe/alpha_g_chain.out` and the 8 top-level files to
`./`, and the 3 files under `moe/repo/` that are not in git at f49a213 to
`../calibration/`.

15 of those 406 are not in git: Triton's `cuda_utils.cpython-312-x86_64-linux-gnu.so`,
one per Triton cache under `MIH4X24CEAJDWYGXCZ4EKW6DTWSUG2ZBSBPV7NGI62HAUKJ7F7BQ/`
(5 counter pages, the census and 9 R3 run directories), which the repository's
`.gitignore` drops by its `*.so` rule (its `results/published/` block
re-includes `.ptx`, `.cubin`, `.ncu-rep`, `.nsys-rep` and `.qdrep`, not `.so`).
The 6 `.ncu-rep` files (five pages and the census) are tracked through
`!results/published/**/*.ncu-rep`. So 391 tarball files are tracked here,
beside `../KIND`, this README, `../calibration/README.md`,
`EXFIL-NOT-PUBLISHED.txt` and the 2 files in `drivers/`.

The other 4615 tarball files sit under the two parts of the checkout the exfil
took, `moe/repo/results/` and `moe/repo/moe/bench/hardware/`, and each is
byte-identical to f49a213 (`git hash-object` on the extracted file equals `git
rev-parse f49a213:<path>`). `EXFIL-NOT-PUBLISHED.txt` lists those and the 15
`.so` files, 4630 in all, one per line with the reason.

The laptop's exfil step also copied `autopilot.log`, `autopilot.out`,
`autopilot.state`, `driver580.log` and `locked_r3.out` off the VM on their own.
They are in the tarball too, and each separate copy is byte-identical to the
tarball's; the tarball's are the ones published.

Every text file published here was searched for credentials (tokens, API keys,
secrets, authorization headers, private key blocks) and none was found. The
VM's hostname, `192-222-53-184`, is in `alpha_g-*/chain.lock` and in the
provenance block of the yaml and every report.
