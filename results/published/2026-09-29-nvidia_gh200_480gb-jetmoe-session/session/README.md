# The held-out test, third of four: JetMoE-8B on one NVIDIA GH200 480GB (Lambda Cloud), run unattended (docs/LAMBDA.md section 3c with `--model jetmoe-8b`); every measurement here is this card's, never averaged with another card's

This session scored the predictions registered in
`docs/registered/2026-09-29-jetmoe-8b-gh200.json` (commit 0f77622, pushed at
a1b6118) before any page here existed: JetMoE-8B (8 experts, top 2, Mixtral's
design at another shape; a CTA runs 32 k-steps on w1 and 88 on w2) predicted
from Mixtral 8x7B's fit on the 2026-09-27 GH200, with the dead-CTA term, the
capacity rule for later re-reads and the per-CTA fixed cost, nothing fitted on
JetMoE. The registration named time from the model's own counted bytes as the
primary test. The same board as the Qwen1.5 and Phi sessions before it:
`board d67185` in `PREFLIGHT.txt`.

## Where it came from

Lambda Cloud instance 9b72732f852b4cd080dbb2681668d2de, gpu_1x_gh200,
us-east-3, active 2026-09-29 22:30:50 UTC, terminated 2026-09-30 00:21:33 UTC
(1.8 h); the id and times are from the laptop's `~/.lambda/rent-ledger.log`.
Driver 570.148.08 upgraded to 580.105.08 and rebooted before setup
(`driver580.log`, `setup_vm.log`). `scripts/gh200_model_session.sh --setup
--model jetmoe-8b` ran at commit a1b6118 (`commit.txt`). This directory is the
branch `run-gh200-jetmoe8b-20260929t2230` at 5372889, curated as the earlier
sessions were; `SHA256SUMS.run-branch` (all 517 files under vm/) and
`PUSHES.run-branch.txt` are its manifest and push log. Preflight PF1 to PF7
PASS (`PREFLIGHT.txt`).

## What ran (`gh200-driver/status`)

| step | UTC | minutes | exit | what |
|---|---|---|---|---|
| prelude | 22:35 | 0 | 0 | persistence, 700 W, the registered locks |
| bytes | 22:35 | 15 | 0 | eight byte pages at the 1710 lock, G = 1 2 4 16 8 32 3 64, treads 1 to 9, 110 to 111 s each, each pushed as it landed; every page INVALID on V10 (G=1 also V6, G = 32 and 64 also V7; below) |
| calibrate | 22:50 | 0 | 0 | the ruler stands (triad 3725.7 GB/s): `../calibration/` |
| timed | 22:51 | 36 | 0 | locked R3 at 1710: G=8, G=32 (treads 6), G=3 (treads 8), every page VALID, worst cell 1710.0 MHz |
| floor | 23:27 | 1 | 3 | floor counters G=64 at base (`r3f-g64.json`) and at the 1710 lock (`r3f-g64-lock1710.json`, FL1 FAIL) |
| deep | 23:28 | 18 | 1 | locked R3 at 1710 to tread 9: G=4 VALID; G=2 slipped (a timed cell at 1680 MHz, 28 cells, run `ff64b86e`, no report) |
| recovery | 23:49 | 30 | 0 | `recovery/retake_auto.sh`: G=2 again on the ladder 1710, 1605, 1500: slipped at 1710 again (1665 MHz, killed after 188 cells, run `a3acb055`, no report), held at 1605 (`1cb74c6f`, VALID at 1605) |

**The clock.** Every byte page fails V10: w1 fits 1676 to 1688 MHz, and w2
1645 to 1662 on the six pages that fit it off the lock; the floor's lock
capture fails FL1 the same way (w1 1683, w2 1641, one cell at 1634). The deep
step's G=2 slipped twice at 1710, the only slips of the queue. Byte and cycle
counts do not depend on the clock; the timed pages read their clock per cell
from NVML and every scored cell held 1710.0 MHz. The module's power cap (0x4)
is in 1 to 7% of the timed ladders' `smi.csv` rows.

## What it answered (registered at 0f77622, scored on these pages)

Sign convention: predicted / measured - 1.

- **Time, the primary test** (`scripts/cross_model_score.py`, 8x7B's fit held,
  JetMoE's own counted bytes from `../results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/`,
  the four VALID lock-1710 pages 593be327 1f286b22 9b196085 d9bd231b; G=2 has
  none at 1710 and its 1605 page is not scored): 58 SHARED and PRIVATE cells,
  rms 7.17%, worst -20.39%, 12 beyond 5% (registered: at or under 2%, none
  beyond 5%). FALSIFIED. SHARED 7.69%, PRIVATE 6.60%, NATIVE (printed) 5.22%.
  The misses are two treads: at n = 1 all eight SHARED and PRIVATE cells are
  predicted fast by 14.6 to 20.4%, at n = 3 the four SHARED cells slow by 10.2
  to 11.0% (PRIVATE at n = 3 within 2%). By tread, all arms: n = 1 14.9% rms,
  n = 2 3.4%, n = 3 8.8%, n = 4 to 9 0.3 to 1.0%.
- **The floor** (the slope of `sm__cycles_elapsed.avg` over CTA k-steps per SM,
  NATIVE G=64, n = 2, 3, 4, 6): w1 360.3 cycles per CTA k-step at base and
  359.8 on the lock capture (registered 360.4: -0.02%, -0.16%): HELD. w2 368.0
  and 368.6 (registered 355.2: +3.60%, +3.78%): FALSIFIED as registered. w2's
  cells run 0.97, 1.45, 1.94 and 2.91 waves (live CTAs / (132 x occupancy 4));
  every GEMM with at least 4 waves a cell reads within 1.3% on the seven
  measured models, and the estimator was corrected for later pages
  (docs/registered/README.md, a156392). The registered result stands.
- **The G >= 8 SHARED slope** over treads 2 to 6: 0.1135 (G=8) and 0.1105
  (G=32) ms per tread against 0.1076 registered (+5.44%, +2.63%). FALSIFIED on
  both; the window holds the n = 2 and 3 cells that miss in time.
- **PRIVATE bytes** (72 cells a GEMM on the eight byte pages): w1 0.19% rms,
  worst +0.58%; w2 0.59%, worst -2.69%; no cell beyond 5%. HELD.
- **Printed, not scored:** time from predicted bytes 7.16% rms (worst -20.37%);
  SHARED bytes w1 20.1% rms (worst -62.1%), w2 11.0% (worst -24.8%).

What the misses point at is in docs/FINDINGS.md (the 2026-09-29/30 held-out
queue); none of it is fitted here.

## Files

- `gh200-driver/status`: the driver's ledger, with the RECOVERY lines;
  `gh200-driver/summary.txt`
- `logs/`: every capture's and run's console
- `locked-r3/<tag>/`: each locked_r3.py ladder's ledger, `smi.csv` and summary
  (`-deepb` is the recovery's G=2 ladder)
- `census-jetmoe-8b.json` and its profiles: the model's census
- `recovery/retake_auto.sh`: the retake step as it ran (it decides from the
  driver's ledger which timed pages to take again)

The `card:` lines of locked_r3.py's ledgers, every page's CARD block and
`PREFLIGHT.json` carry the card's UUID, as every published session does.
