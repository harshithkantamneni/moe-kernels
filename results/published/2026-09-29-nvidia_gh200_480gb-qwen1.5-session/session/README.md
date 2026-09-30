# The held-out test, first of four: Qwen1.5-MoE-A2.7B on one NVIDIA GH200 480GB (Lambda Cloud), run unattended (docs/LAMBDA.md section 3c with `--model qwen1.5-moe-a2.7b`); every measurement here is this card's, never averaged with another card's

This session scored the predictions registered in
`docs/registered/2026-09-29-qwen1.5-moe-a2.7b-gh200.json` (commit 0f77622,
pushed at a1b6118) before any page here existed: Qwen1.5-MoE-A2.7B (60
experts, top 4; a CTA runs 32 k-steps on w1 and 22 on w2) predicted from
Mixtral 8x7B's fit on the 2026-09-27 GH200, with the dead-CTA term, the
capacity rule for later re-reads and the per-CTA fixed cost, nothing fitted on
Qwen1.5. The registration named time from the model's own counted bytes as the
primary test. A sixth GH200 board: `board d67185` in `PREFLIGHT.txt`. The
Phi-3.5-MoE and JetMoE-8B sessions of the same queue ran on this board too
(Lambda gave the same machine three times, at the same address).

## Where it came from

Lambda Cloud instance dc9d0aa1880445079e8502bf01a4685c, gpu_1x_gh200,
us-east-3, active 2026-09-29 18:12:13 UTC, terminated 19:54:57 UTC (1.7 h);
the id and times are from the laptop's `~/.lambda/rent-ledger.log`. Driver
570.148.08 upgraded to 580.105.08 and rebooted before setup (`driver580.log`,
`setup_vm.log`). `scripts/gh200_model_session.sh --setup --model
qwen1.5-moe-a2.7b` ran at commit a1b6118 (`commit.txt`), which carries the
registration. This directory is the branch
`run-gh200-qwen15moea27b-20260929t1812` at f173c68, curated as the earlier
sessions were; `SHA256SUMS.run-branch` (all 473 files under vm/) and
`PUSHES.run-branch.txt` are its manifest and push log. Preflight PF1 to PF7
PASS (`PREFLIGHT.txt`).

## What ran (`gh200-driver/status`)

| step | UTC | minutes | exit | what |
|---|---|---|---|---|
| prelude | 18:16 | 0 | 0 | persistence, 700 W, the registered locks |
| bytes | 18:16 | 23 | 0 | eight byte pages at the 1710 lock, G = 1 2 4 16 8 32 3 64, treads 1 to 9, 168 to 169 s each, each pushed as it landed; every page INVALID on V10 (below) |
| calibrate | 18:40 | 0 | 0 | the ruler stands (triad 3726.1 GB/s): `../calibration/` |
| timed | 18:40 | 36 | 0 | locked R3 at 1710: G=8, G=32 (treads 6), G=3 (treads 8), every page VALID, worst cell 1710.0 MHz |
| floor | 19:16 | 1 | 3 | floor counters G=64 at base (`r3f-g64.json`) and at the 1710 lock (`r3f-g64-lock1710.json`, FL1 FAIL, below) |
| deep | 19:17 | 33 | 0 | locked R3 at 1710 to tread 9: G=4 and G=2 VALID |
| recovery | 19:53 | 0 | 0 | `recovery/retake_auto.sh`: nothing to retake |

**V10 and FL1.** Every byte page fails V10, the gate that reads each GEMM's
clock off its own counters (`r3_lock_fit`): w2 fits 1686 to 1690 MHz on every
page while w1 fits 1699 to 1703 (the G=16 page also names one cell at 1666),
and the floor's lock capture fails FL1 on the same pattern (w2 1680 MHz, w1
1714). The same constant gap between the two GEMMs of one call as on OLMoE;
byte counts and cycle counts do not depend on it. Every page except G=1 also
fails C1 and G=1 fails C3 (claims, not validity). The timed pages read their
clock per cell from NVML and all held 1710.0 MHz; the module's power cap (0x4)
is in 3 to 4% of their `smi.csv` rows.

## What it answered (registered at 0f77622, scored on these pages)

Sign convention: predicted / measured - 1.

- **Time, the primary test** (`scripts/cross_model_score.py`, 8x7B's fit held,
  Qwen1.5's own counted bytes from `../results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/`,
  the five VALID lock-1710 pages 8b67e946 cd42c1ed 89bd3066 00804a3a 176ddae1):
  76 SHARED and PRIVATE cells, rms 1.87%, worst +4.81%, none beyond 5%
  (registered: at or under 2%, none beyond 5%). HELD. SHARED 1.78%, PRIVATE
  1.96%, NATIVE (printed) 0.51%. The residual falls with the tread: 3.79% rms
  at n = 1 (all arms), 1.55% at n = 2, at or under 0.95% from n = 3.
- **The floor** (the slope of `sm__cycles_elapsed.avg` over CTA k-steps per SM,
  NATIVE G=64, n = 2, 3, 4, 6): w1 361.0 cycles per CTA k-step at base and
  361.7 on the lock capture (registered 360.4: +0.17%, +0.37%); w2 385.8 and
  385.5 (registered 388.6: -0.73%, -0.81%). HELD. Every cell runs at least 7.3
  waves (live CTAs / (132 x occupancy)), so the corrected estimator of
  2026-09-29 reads the same numbers.
- **The G >= 8 SHARED slope** over treads 2 to 6: 0.2156 (G=8) and 0.2154 (G=32)
  ms per tread against 0.2159 registered (-0.15%, -0.21%). HELD.
- **PRIVATE bytes** (72 cells a GEMM on the eight byte pages): w1 1.31% rms,
  worst +3.24%; w2 0.53%, worst -2.19%; no cell beyond 5%. HELD.
- **Printed, not scored:** time from predicted bytes 2.12% rms (worst +5.26%);
  SHARED bytes w1 7.4% rms (worst +14.2%), w2 3.9% (worst +10.6%).

Every registered falsifier held. What it means beside the other three models
is in docs/FINDINGS.md (the 2026-09-29/30 held-out queue); nothing is fitted
here.

## Files

- `gh200-driver/status`: the driver's ledger, with the RECOVERY lines;
  `gh200-driver/summary.txt`
- `logs/`: every capture's and run's console
- `locked-r3/<tag>/`: each locked_r3.py ladder's ledger, `smi.csv` and summary
- `census-qwen1.5-moe-a2.7b.json` and its profiles: the model's census
- `recovery/retake_auto.sh`: the retake step as it ran (it decides from the
  driver's ledger which timed pages to take again)

The `card:` lines of locked_r3.py's ledgers, every page's CARD block and
`PREFLIGHT.json` carry the card's UUID, as every published session does.
