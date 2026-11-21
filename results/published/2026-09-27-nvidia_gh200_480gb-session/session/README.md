# One of the study's four cards, the primary one: the GH200 model-test session (docs/LAMBDA.md section 3c), run unattended on one NVIDIA GH200 480GB (Lambda Cloud); every measurement here is this card's, never averaged with another card's

This session tested the two NOT-FINAL models of docs/COUNTERS.md 6.7 on a
GH200: the wave-split byte model (`scripts/wave_split_bytes.py`) and the timing
model LOWM-1/2 (`scripts/r3_timing_model.py`, predictions P1 to P8, registered
from the 2026-09-25 GH200 session before any page here existed). It is a
different board from 2026-09-25's (`board 9b6d01` on every page's CARD line).

## Where it came from

Lambda Cloud instance 641035c6ffc649b382fad0bf897c424c, type gpu_1x_gh200,
region us-east-3, $2.29/h, launched 2026-09-27 05:46 UTC and terminated about
09:47 UTC (about 4.0 h, about $9.2); the instance id and the two times are from
the laptop's `~/.lambda/rent-ledger.log`, not from a file here. The instance
shipped driver 570.148.08; by the owner's decision (2026-09-26) it was upgraded
to `nvidia-driver-580-server-open` 580.105.08-0lambda0.22.04.1 and rebooted
before setup (`driver580.log`, the apt console; `scripts/vm_run.sh prepare`).

Nothing was typed on the VM. `scripts/gh200_model_session.sh --setup` ran
`setup_vm.sh` at commit 161f9ec19347efb7aa64ac4c645c527e0d2b46d6 (`commit.txt`,
`setup_vm.log`), then the session, and after every step
`scripts/vm_results_push.sh` pushed the VM's `~/moe/results` and `~/moe/session`
to the orphan branch `run-gh200-2026-09-27` of this repository, with a deploy
key made on the VM and deleted after the run. This directory is that branch's
last commit, curated: `results/` is the VM's `~/moe/results` (less its
calibration run directory, which is in `../calibration/` with the ruler), and
this directory is `~/moe/session` plus `setup_vm.log`, `driver580.log` and the
driver's console `gh200-driver.out`. `SHA256SUMS.run-branch` is the branch's
manifest (every file as the VM committed it; the laptop checked all 684 lines
before the terminate) and `PUSHES.run-branch.txt` its push log.

Card: NVIDIA GH200 480GB, sm_90, 132 SMs, 60 MiB L2 (`PREFLIGHT.json`), 97871
MiB, Ubuntu 22.04 aarch64, 525 GB RAM, driver 580.105.08, CUDA 13.0
(`setup_vm.log` S0). Nsight Compute at `/usr/local/cuda-13.0/bin/ncu`
(`ncu.txt`). Counter door `sudo` (`setup_vm.log`). Preflight PF1 to PF7 PASS
(`PREFLIGHT.txt`).

## What ran, in order (`gh200-driver/status`, one UTC line per event)

| step | UTC | minutes | exit | what |
|---|---|---|---|---|
| prelude | 05:52 | 0 | 0 | supported clocks (1710, 1605, 1500, 1410 all supported: `lock-plan.txt`), persistence on, GPU power limit 700 W at `-sc 0` (`power-at-start.txt`, `power-set.txt`) |
| bytes | 05:53 | 12 | 3 | byte pages at the 1710 MHz lock, treads 1 to 9: G=1 written and INVALID on V10 only, so the loop stopped by its rule; the G=2 base-clock control (`results/.../base/r3c-g2.json`) passed every gate |
| calibrate | 06:05 | 0 | 0 | the ruler stands (triad 3726.1 GB/s, `not_throttled` PASS at 1470 MHz): `../calibration/measured_nvidia_gh200_480gb.yaml` |
| timed | 06:05 | 36 | 0 | locked R3 at 1710: G=8 and 32 (treads 6), G=3 (treads 8); every page VALID, worst cell 1710.0 MHz |
| eta | 06:41 | 43 | 0 | locked R3 at 1410 (G=4, 2), 1500 (G=4), 1605 (G=4); every lock held, every page VALID |
| floor | 07:24 | 4 | 3 | floor counters G=64 and G=2 at base, G=64 unlocked (a record), G=64 at the 1710 lock: the last INVALID on FL1 only |
| deep | 07:28 | 32 | 0 | locked R3 at 1710 to tread 9, G=4 and G=2: both VALID, 9-copy declaration |
| r1lock | 08:00 | 53 | 0 | R1 in lock mode at 1710 1500 1410, G=4 and G=1: every VALIDITY gate PASS |
| recovery | 09:02 | 43 | 0 | the lock-1710 byte pages the loop never took (G = 2 4 16 8 32 3 64), run by hand after DRIVER-DONE: `recovery/lock_bytes.sh`, the script exactly as it ran; its lines are in the same ledger |

## The V10 and FL1 failures were the gate, not the clock

Every lock page (`results/2026-09-27-nvidia_gh200_480gb-r3-counters/lock1710/`)
and the floor at the lock failed V10 or FL1 as captured, on a lock that held.
Those gates held each cell's own `sm__cycles_elapsed.avg` over its
`gpu_time_ns` to 1710 MHz, and ncu's duration carries a fixed overhead its
cycle count does not: on the G=1 page, over its 54 GEMMs, duration = 22.7 us +
cycles / 1705 MHz, so the per-cell ratio read 1632 MHz on the 0.28 ms kernels
and 1697 on the 4.8 ms ones. Every page fits f = 1694 to 1712 MHz with an
offset of 15 to 24 us. The gates now fit the clock through the overhead
(`r3_lock_fit`, commit f97df00, branch gh200-analysis-2026-09-27). The pages
here are as the VM wrote them; re-scored with `--analyse` at f97df00:

| lock page | as captured | re-scored |
|---|---|---|
| G=1, 3, 4, 8, 16 | V10 FAIL | every VALIDITY gate PASS |
| G=32, 64 | V7, V10 FAIL | V7 FAIL alone: the launch-order case expected at G >= 32 (NATIVE's narrower grid) |
| G=2 | V7, V10 FAIL | V7 (one cell, n=2 w2, NATIVE and SHARED 1.33% apart against 1.0%, the cell's own calls 1.03% apart) and V10 (w2 fitted at 1694 MHz, one step and 1 MHz under; the timed G=2 page's NVML also read 1695) |
| floor G=64 at the lock | FL1 FAIL | FL1 PASS (w1 1717.2, w2 1701.2 MHz) |

## What it answered (registered 2026-09-25, scored on these pages)

- P1 held locks: the G=4 SHARED slope 0.6598, 0.6222, 0.5812 ms per tread at
  1410, 1500, 1605 MHz, against 0.6654, 0.6259, 0.5854 at eta 1 and 0.5878,
  0.5754, 0.5621 at eta 0.35; the G=2 steps at 1410 (0.541 0.732 0.591 0.731
  0.595) are eta 1's (0.550 0.731 0.600 0.733 0.598). HELD, eta 1.
- R1 lock mode: elasticity 0.988 [0.985, 0.990] at G=4, 0.312 [0.311, 0.313]
  at G=1.
- P2: G=8 and G=32 SHARED steps 2 to 6 all inside [0.538, 0.556]. HELD.
- P3: G=2's odd-n excess over G=4 at n = 3 5 7 9: 0.154 0.153 0.162 0.160 ms
  (registered 0.143 0.145 0.144 0.151; falsifier below 0.10 or trending). HELD.
- P4: G=4 n=8 to 9, 0.508 ms against 0.504 registered. HELD.
- P5: G=3 was registered flat; its steps n=2 to 8 read 0.491 0.576 0.550 |
  0.516 0.580 0.560, a period-3 ripple. FALSIFIED.
- P6: the floor's own counters read 350 to 352 cycles per CTA k-step on w1
  and 354 to 367 on w2, the same at ncu's base clock, unlocked and at the
  1710 lock (registered 353.8; the PTX's shared-memory traffic is 352 at 128
  B/clk); the shared-memory pipe at 75 to 83% of peak. HELD.
- Bytes: w2 SHARED at G=8, n=2 and 4: 1.078 and 1.100, against WSC 1.082 and
  1.118 and the LRU rival's 1.139 and 1.159. HELD. Every w1 and PRIVATE cell
  within 1.5% except w1 SHARED G=64 (up to 3.9%). w2 SHARED at n >= 5 is
  under-predicted by more than the registered 5% at G=16 (7.5, 12.6, 16.3%),
  G=4 n=8 (5.9%) and G=2 (6.0, 6.2, 13.2%, on the page that fails V7):
  FALSIFIED there.
- P7 and P8 are not answered by this session (docs/LAMBDA.md section 3c).

## Files

- `gh200-driver/status`: the driver's ledger; `gh200-driver/summary.txt`
- `logs/`: every capture's and run's console (`driver-<step>.log` per step)
- `locked-r3/<tag>/`: each locked_r3.py ladder's ledger and summary
- `supported-clocks.txt`, `lock-plan.txt`, `power-*.txt`, `clocks-*.txt`: the
  card's state around each step
- `census.json`, `census.profiles/`: the preflight's census (PF6)
- `recovery/lock_bytes.sh`: the recovery run by hand after the driver

The `card:` lines of locked_r3.py's ledgers, every page's CARD block and
`PREFLIGHT.json` carry the card's UUID, as every published session does.
