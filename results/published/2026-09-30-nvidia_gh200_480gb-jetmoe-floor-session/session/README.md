# A floor-only session: JetMoE-8B on one NVIDIA GH200 480GB, the floor capture at 4+ waves (docs/registered/README.md, 2026-09-30)

JetMoE-8B's w2 (8 experts x 32 N-tiles) runs 0.485 x n waves, so its first floor capture (n = 2, 3, 4, 6, in `2026-09-29-nvidia_gh200_480gb-jetmoe-session`) had no w2 cell of 4 or more waves and its w2 floor, read as the slope over all four, came out 368.0 against 355.2 (FALSIFIED as registered). This session takes the capture at `dram_counter_route.r3_floor_treads`'s n = 2, 3, 4, 6, 9, 10, 11, and the registration (79c5034) scores w2 on n = 9, 10, 11 (4.4 to 5.3 waves) and w1 on every cell.

## Where it came from

Lambda Cloud instance 82535565571843859e1e4bb3dadeb6c3, gpu_1x_gh200, us-east-3, active 2026-09-30 06:03:15 UTC,
terminated 06:10:34 UTC by the laptop's guard once every pushed file had been
verified (126 files match SHA256SUMS); ids and times from the laptop's
`~/.lambda/rent-ledger.log`. Board `594c0f` (`PREFLIGHT.txt`), driver 570.148.08
upgraded to 580.105.08 before setup (`driver580.log`). `scripts/gh200_model_session.sh
--setup --model jetmoe-8b --steps prelude,floor` ran at commit e6b1e20
(`commit.txt`), which carries the registration. This directory is the branch
`run-gh200-floor-jetmoe8b-20260930t0603` at 3d7b96e; `SHA256SUMS.run-branch` and `PUSHES.run-branch.txt` are
its manifest and push log.

## What ran (`gh200-driver/status`)

Prelude (06:06, exit 0), then the floor step (06:06 to 06:07): `r3f-g64.json` at ncu's base clock (every gate PASS) and `r3f-g64-lock1710.json` at the 1710 MHz lock (FL1 FAIL: w1 fitted 1695.5 MHz, w2 1669.1 MHz). The r3f-g2 and unlocked captures are skipped for a model no prediction reads them for. Nothing to retake.

The lock capture fails FL1 alone, as every short-kernel lock capture of the
study has (the clock fit through ncu's fixed overhead reads w1 and w2 a
constant gap apart, which one clock cannot make); its counts agree with the
base capture's. Every number below is the slope of `sm__cycles_elapsed.avg`
over CTA k-steps per SM (launch grid x k-steps / 132), on the cells the
registration names.

## What it answered (registered at 79c5034, before any page)

| GEMM | cells | registered | base capture | lock capture |
|---|---|---|---|---|
| w1 (32 k-steps) | n = 2, 3, 4, 6, 9, 10, 11 | 360.4 | 360.8 (+0.12%) | 361.9 (+0.43%) |
| w2 (88 k-steps) | n = 9, 10, 11 | 355.2 | 361.7 (+1.84%) | 364.5 (+2.61%) |

HELD on the base capture, both GEMMs inside 2%; the w2 all-cell line reads 367.2, the estimator bias the registration corrects. w2 sits on the high side: at 4+ waves the per-CTA floor, 344.1 + 979 / 88, is 1.8% under JetMoE's measured w2. The lock capture, which fails FL1, reads w2 at +2.61%.

## Files

- `gh200-driver/status`, `logs/driver-floor.log`, `logs/r3f-*.log`: the ledger and consoles
- `results/2026-09-30-nvidia_gh200_480gb-r3-counters/r3f-g64*.json`: the floor captures
- `recovery/retake_auto.sh`: the retake step as it ran (nothing to retake)

The `card:` lines and every page's CARD block carry the card's UUID, as every
published session does.
