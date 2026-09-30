# A floor-only session: Mixtral 8x7B on one NVIDIA GH200 480GB, the floor capture at 4+ waves (docs/registered/README.md, 2026-09-30)

Mixtral 8x7B's w2 (64 N-tiles) runs 0.97 x n waves, so its floor captures at n = 2, 3, 4, 6 had one w2 cell of 4 or more waves; its w2 all-cell slope read about 362 (4% over the per-CTA floor). This session takes the capture at n = 2, 3, 4, 6, 7, 8, and the registration (79c5034) scores w2 on n = 6, 7, 8 (5.8 to 7.8 waves) and w1 on every cell. 8x7B's earlier floor counters are among the calibration data for c and F; the cells at n = 7 and 8 are new.

## Where it came from

Lambda Cloud instance 8be8788495dc45a383ce6a2765e77581, gpu_1x_gh200, us-east-3, active 2026-09-30 06:14:48 UTC,
terminated 06:26:15 UTC by the laptop's guard once every pushed file had been
verified (153 files match SHA256SUMS); ids and times from the laptop's
`~/.lambda/rent-ledger.log`. Board `594c0f` (`PREFLIGHT.txt`), driver 570.148.08
upgraded to 580.105.08 before setup (`driver580.log`). `scripts/gh200_model_session.sh
--setup --model mixtral-8x7b --steps prelude,floor` ran at commit e6b1e20
(`commit.txt`), which carries the registration. This directory is the branch
`run-gh200-floor-mixtral8x7b-20260930t0614` at 84c7c4a; `SHA256SUMS.run-branch` and `PUSHES.run-branch.txt` are
its manifest and push log.

## What ran (`gh200-driver/status`)

Prelude (06:17, exit 0), then the floor step (06:17 to 06:22): `r3f-g64.json` at ncu's base clock, `r3f-g2.json`, `r3f-g64-unlocked.json` (a record at the unlocked clock) and `r3f-g64-lock1710.json` at the 1710 MHz lock (FL1 FAIL). Nothing to retake.

The lock capture fails FL1 alone, as every short-kernel lock capture of the
study has (the clock fit through ncu's fixed overhead reads w1 and w2 a
constant gap apart, which one clock cannot make); its counts agree with the
base capture's. Every number below is the slope of `sm__cycles_elapsed.avg`
over CTA k-steps per SM (launch grid x k-steps / 132), on the cells the
registration names.

## What it answered (registered at 79c5034, before any page)

| GEMM | cells | registered | base capture | unlocked | lock capture |
|---|---|---|---|---|---|
| w1 (64 k-steps) | n = 2, 3, 4, 6, 7, 8 | 352.2 | 349.9 (-0.66%) | 349.9 (-0.64%) | 350.1 (-0.60%) |
| w2 (224 k-steps) | n = 6, 7, 8 | 348.5 | 354.2 (+1.65%) | 347.3 (-0.33%) | 348.7 (+0.06%) |

HELD on every capture, both GEMMs inside 2%. The w2 all-cell line reads 356.0 to 357.9, the few-wave bias; on its 4+-wave cells 8x7B's w2 floor is the per-CTA floor to 1.7%.

## Files

- `gh200-driver/status`, `logs/driver-floor.log`, `logs/r3f-*.log`: the ledger and consoles
- `results/2026-09-30-nvidia_gh200_480gb-r3-counters/r3f-g64*.json`: the floor captures
- `recovery/retake_auto.sh`: the retake step as it ran (nothing to retake)

The `card:` lines and every page's CARD block carry the card's UUID, as every
published session does.
