# The held-out test, second of four: Phi-3.5-MoE on one NVIDIA GH200 480GB (Lambda Cloud), run unattended (docs/LAMBDA.md section 3c with `--model phi-3.5-moe`); every measurement here is this card's, never averaged with another card's

This session scored the predictions registered in
`docs/registered/2026-09-29-phi-3.5-moe-gh200.json` (commit 0f77622, pushed at
a1b6118) before any page here existed: Phi-3.5-MoE (16 experts, top 2; a CTA
runs 64 k-steps on w1 and 100 on w2) predicted from Mixtral 8x7B's fit on the
2026-09-27 GH200, with the dead-CTA term, the capacity rule for later re-reads
and the per-CTA fixed cost, nothing fitted on Phi. The registration named time
from the model's own counted bytes as the primary test. The same board as the
Qwen1.5-MoE-A2.7B session before it: `board d67185` in `PREFLIGHT.txt`.

## Where it came from

Lambda Cloud instance 951662df343b46f8b8c21c864c504ca4, gpu_1x_gh200,
us-east-3, active 2026-09-29 20:19:58 UTC, terminated 22:26:49 UTC (2.1 h);
the id and times are from the laptop's `~/.lambda/rent-ledger.log`. Driver
570.148.08 upgraded to 580.105.08 and rebooted before setup (`driver580.log`,
`setup_vm.log`). `scripts/gh200_model_session.sh --setup --model phi-3.5-moe`
ran at commit a1b6118 (`commit.txt`). This directory is the branch
`run-gh200-phi35moe-20260929t2019` at a670527, curated as the earlier sessions
were; `SHA256SUMS.run-branch` (all 473 files under vm/) and
`PUSHES.run-branch.txt` are its manifest and push log. Preflight PF1 to PF7
PASS (`PREFLIGHT.txt`).

## What ran (`gh200-driver/status`)

| step | UTC | minutes | exit | what |
|---|---|---|---|---|
| prelude | 20:25 | 0 | 0 | persistence, 700 W, the registered locks |
| bytes | 20:25 | 46 | 0 | eight byte pages at the 1710 lock, G = 1 2 4 16 8 32 3 64, treads 1 to 9, 330 to 361 s each, each pushed as it landed; G = 2, 3, 4 pass every gate, G=1 and G=64 fail V10, G = 8, 16, 32 end on a claim (below) |
| calibrate | 21:10 | 0 | 0 | the ruler stands (triad 3725.7 GB/s): `../calibration/` |
| timed | 21:11 | 36 | 0 | locked R3 at 1710: G=8, G=32 (treads 6), G=3 (treads 8), every page VALID, worst cell 1695.0 MHz (G=3) |
| floor | 21:47 | 2 | 0 | floor counters G=64 at base (`r3f-g64.json`) and at the 1710 lock (`r3f-g64-lock1710.json`), both passing their gates |
| deep | 21:49 | 33 | 0 | locked R3 at 1710 to tread 9: G=4 and G=2 VALID |
| recovery | 22:25 | 0 | 0 | `recovery/retake_auto.sh`: nothing to retake |

**The byte pages' gates.** V10 fails on G=1 (w2 fitted 1695 MHz) and G=64
(two n = 1 w2 cells at 1762 to 1763 MHz); byte counts do not depend on the
clock. G = 8, 16, 32 exit on C1/C2 REFUSE, and G = 32 and 64 on C6 FAIL and
G=64 on C2 FAIL: claims about the byte model's own survival ladder, not
validity. The timed pages read their clock per cell from NVML; the module's
power cap (0x4) is in 15 to 21% of their `smi.csv` rows, more than on the other
three models of the queue, and the G=3 page's worst cell read 1695 MHz, inside
the one-step band.

## What it answered (registered at 0f77622, scored on these pages)

Sign convention: predicted / measured - 1.

- **Time, the primary test** (`scripts/cross_model_score.py`, 8x7B's fit held,
  Phi's own counted bytes from `../results/2026-09-29-nvidia_gh200_480gb-r3-counters/lock1710/`,
  the five VALID lock-1710 pages d4af2011 29a83ec7 2b704b65 7b62501d dbd2c524):
  76 SHARED and PRIVATE cells, rms 0.57%, worst -1.27%, none beyond 5%
  (registered: at or under 2%, none beyond 5%). HELD. SHARED 0.55%, PRIVATE
  0.59%, NATIVE (printed) 0.64%; no tread above 0.76%.
- **The floor** (the slope of `sm__cycles_elapsed.avg` over CTA k-steps per SM,
  NATIVE G=64, n = 2, 3, 4, 6): w1 349.6 cycles per CTA k-step at base and
  349.8 on the lock capture (registered 352.2: -0.73%, -0.67%); w2 355.4 and
  354.4 (registered 353.9: +0.42%, +0.14%). HELD. w2's n = 2 cell runs 3.9
  waves; on the cells of at least 4 waves (n = 3, 4, 6, the estimator
  corrected on 2026-09-29 for later pages) w2 reads 351.7 and 353.4.
- **The G >= 8 SHARED slope** over treads 2 to 6: 0.4839 (G=8) and 0.4828
  (G=32) ms per tread against 0.4898 and 0.4896 registered (-1.21%, -1.39%).
  HELD.
- **PRIVATE bytes** (72 cells a GEMM on the eight byte pages): w1 0.36% rms,
  worst -1.94%, none beyond 5%: HELD. w2 1.81% rms, worst -7.99%, 3 cells
  beyond 5%, all at G=64: n = 7 -5.14%, n = 8 -7.99%, n = 9 -6.83% (registered:
  every cell within 5%). FALSIFIED on w2, at the deepest G=64 cells only; the
  G=64 byte page is the one that fails V10 and C2.
- **Printed, not scored:** time from predicted bytes 0.65% rms (worst -2.48%);
  SHARED bytes w1 10.8% rms (worst -42.7%), w2 18.5% (worst -60.3%): an
  expert's weights fit in the L2, the region with no calibrated survival law.

What it means beside the other three models is in docs/FINDINGS.md (the
2026-09-29/30 held-out queue); nothing is fitted here.

## Files

- `gh200-driver/status`: the driver's ledger, with the RECOVERY lines;
  `gh200-driver/summary.txt`
- `logs/`: every capture's and run's console
- `locked-r3/<tag>/`: each locked_r3.py ladder's ledger, `smi.csv` and summary
- `census-phi-3.5-moe.json` and its profiles: the model's census
- `recovery/retake_auto.sh`: the retake step as it ran (it decides from the
  driver's ledger which timed pages to take again)

The `card:` lines of locked_r3.py's ledgers, every page's CARD block and
`PREFLIGHT.json` carry the card's UUID, as every published session does.
