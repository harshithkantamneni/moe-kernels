# The small-K test: OLMoE-1B-7B on one NVIDIA GH200 480GB (Lambda Cloud), run unattended (docs/LAMBDA.md section 3c with `--model olmoe-1b-7b`); every measurement here is this card's, never averaged with another card's

This session scored the predictions registered in
`docs/registered/2026-09-29-olmoe-1b-7b-gh200.json` (commit 7d9a1a1) before any
page here existed: OLMoE-1B-7B (64 experts, top 8; a CTA runs 32 k-steps on w1
and 16 on w2) predicted from Mixtral 8x7B's fit on the 2026-09-27 GH200, with
the timing model's dead-CTA term and the byte model's capacity rule for later
re-reads, nothing fitted on OLMoE. The registration named time from OLMoE's own
counted bytes as the primary test. A fifth GH200 board: `board d663f7` in
`PREFLIGHT.txt`. The same test was also registered from the H100 SXM5's fit
(`2026-09-29-olmoe-1b-7b-h100.json`); the GH200 was available first and that
registration was not run.

## Where it came from

Lambda Cloud instance ae8f9dde0b3e4ad9978d34bb2d3154c7, gpu_1x_gh200,
us-east-3, $2.29/h, active 2026-09-29 11:57:11 UTC, terminated 13:31:22 UTC
(1.6 h, about $3.6) by the laptop's guard once every pushed file had been
verified (all 473 files under vm/ match SHA256SUMS); the id and times are from
the laptop's `~/.lambda/rent-ledger.log`. Driver 570.148.08 upgraded to
580.105.08 and rebooted before setup (`driver580.log`, `setup_vm.log`).
`scripts/gh200_model_session.sh --setup --model olmoe-1b-7b` ran at commit
b87d547 (`commit.txt`), which carries both registrations. This directory is the
branch `run-gh200-olmoe-20260929t1157` at c0eb796, curated as the earlier
sessions were; `SHA256SUMS.run-branch` and `PUSHES.run-branch.txt` are its
manifest and push log. Preflight PF1 to PF7 PASS (`PREFLIGHT.txt`).

## What ran (`gh200-driver/status`)

| step | UTC | minutes | exit | what |
|---|---|---|---|---|
| prelude | 12:00 | 0 | 0 | persistence, 700 W, the registered locks |
| bytes | 12:00 | 19 | 0 | eight byte pages at the 1710 lock, G = 1 2 4 16 8 32 3 64, treads 1 to 9, 140 to 141 s each, each pushed as it landed; every page INVALID on V10 alone (below) |
| calibrate | 12:20 | 0 | 0 | the ruler stands: `../calibration/` |
| timed | 12:20 | 36 | 0 | locked R3 at 1710: G=8, G=32 (treads 6), G=3 (treads 8), every page VALID, worst cell 1710.0 MHz |
| floor | 12:56 | 1 | 3 | floor counters G=64 at base (`r3f-g64.json`) and at the 1710 lock (`r3f-g64-lock1710.json`, FL1 FAIL on its fit, below) |
| deep | 12:57 | 32 | 0 | locked R3 at 1710 to tread 9: G=4 and G=2 VALID |
| recovery | 13:29 | 0 | 0 | `recovery/retake_auto.sh`: nothing to retake |

**V10 and FL1 on short kernels.** Every byte page and the floor's lock
capture fail only the gate that reads each GEMM's clock off its own counters,
duration = t0 + cycles / f fitted over the page's cells (`r3_lock_fit`): w1
fits 1686 to 1689 MHz and w2 1662 to 1668 MHz on every byte page (the floor's
lock capture 1702 and 1666). One clock would read the same on both GEMMs of a
call; a constant gap between them points at the fit, whose fixed ncu duration
overhead is a large share of OLMoE's tens-of-microsecond GEMMs. Not settled
here. Byte counts do not depend on the clock. The
timed pages read their clock per cell from NVML and all held 1710.0 MHz; the
module's power cap (0x4) is in 8 to 10% of their `smi.csv` rows.

## What it answered (registered at 7d9a1a1, scored on these pages)

- **Time, the primary test** (`scripts/cross_model_score.py`, 8x7B's fit held,
  OLMoE's own counted bytes, the five VALID pages): 76 SHARED and PRIVATE
  cells, rms 3.53%, worst -5.81%, 12 beyond 5% (registered: at or under 2%,
  none beyond 5%). FALSIFIED. SHARED 4.54% rms and fast, PRIVATE 2.07%, NATIVE
  (printed) 5.64%.
- **The G >= 8 SHARED slope** over treads 2 to 6: 0.1705 (G=8) and 0.1707
  (G=32) ms per tread against 0.1597 registered, band 0.1565 to 0.1629.
  FALSIFIED, 6.8% steep.
- **The floor on w1**: the slope of `sm__cycles_elapsed.avg` over CTA k-steps
  per SM across n = 2, 3, 4, 6 reads 360.5 cycles at base and 359.4 on the lock
  capture (registered 340 to 365). HELD. w2 reads 401.6 and 401.7, against 354 to
  367 on 8x7B (224 k-steps a CTA) and 369 on Qwen2-57B (40): the floor per
  k-step rises as the CTA's k-steps fall.
- **PRIVATE bytes**: w1 1.97% rms, worst +4.32%; w2 0.30%, worst -1.53%; no cell
  beyond 5% (registered: every cell within 5%). HELD.
- **Printed, not scored:** time from predicted bytes 3.83% rms; SHARED bytes
  w1 9.8% rms (worst +19.0%), w2 2.9% (worst +7.8%), the first measurements of L2
  survival at 0.07 to 0.3 of the L2 in reuse distance.

What the misses point at is in docs/FINDINGS.md (2026-09-29, the small-K
test); none of it is fitted here.

## Files

- `gh200-driver/status`: the driver's ledger, with the RECOVERY lines;
  `gh200-driver/summary.txt`
- `logs/`: every capture's and run's console
- `locked-r3/<tag>/`: each locked_r3.py ladder's ledger, `smi.csv` and summary
- `census-olmoe-1b-7b.json` and its profiles: the model's census
- `recovery/retake_auto.sh`: the retake step as it ran (it decides from the
  driver's ledger which timed pages to take again)

The `card:` lines of locked_r3.py's ledgers, every page's CARD block and
`PREFLIGHT.json` carry the card's UUID, as every published session does.
