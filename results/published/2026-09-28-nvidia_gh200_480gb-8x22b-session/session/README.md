# The cross-model test: Mixtral 8x22B on one NVIDIA GH200 480GB (Lambda Cloud), run unattended (docs/LAMBDA.md section 3c with `--model mixtral-8x22b`); every measurement here is this card's, never averaged with another card's

This session scored the predictions registered in
`docs/registered/2026-09-27-mixtral-8x22b-gh200.json` before any page here
existed: Mixtral 8x22B's bytes and call times, predicted from the fit of
Mixtral 8x7B on the 2026-09-27 GH200 session with nothing fitted on 8x22B
(`scripts/cross_model_predict.py`). It is a third GH200 board: `board 435984`
on every page's CARD line and in `PREFLIGHT.txt` (2026-09-25 and 2026-09-27
were other boards).

## Where it came from

Lambda Cloud instance 3cc7afea64e04737ba15c07ea479ebd5, type gpu_1x_gh200,
region us-east-3, launched 2026-09-28 01:44:56 UTC and terminated 04:53:30 UTC
(about 3.1 h, about $7.2 at $2.29/h); the id and the times are from the
laptop's `~/.lambda/rent-ledger.log`, not from a file here. The instance
shipped driver 570.148.08; as on 2026-09-27 it was upgraded to 580.105.08 and
rebooted before setup (`driver580.log`, `setup_vm.log`: driver 580.105.08,
CUDA 13.0).

Nothing was typed on the VM except one operator line (below).
`scripts/gh200_model_session.sh --setup --model mixtral-8x22b` ran
`setup_vm.sh` at commit f45358d997070ab5a93cc5a745efd1ba81aa52ed
(`commit.txt`), then the session, and after every step
`scripts/vm_results_push.sh` pushed `~/moe/results` and `~/moe/session` to
the orphan branch `run-gh200-8x22b-2026-09-28`, with a deploy key made on the
VM and deleted after the run. This directory is that branch's last commit
(35c55a3), curated as 2026-09-27's was: `results/` is `~/moe/results` less
its calibration run directory, which is in `../calibration/` with the ruler;
this directory is `~/moe/session` plus `setup_vm.log`, `driver580.log` and
the driver's console `gh200-driver.out`. `SHA256SUMS.run-branch` is the
branch's manifest (the laptop checked all 543 lines) and
`PUSHES.run-branch.txt` its push log; `HELD-BACK.txt` was empty.

Card: NVIDIA GH200 480GB, sm_90, 132 SMs, 60 MiB L2 (`PREFLIGHT.txt`).
Preflight PF1 to PF7 PASS (`PREFLIGHT.txt`).

## What ran, in order (`gh200-driver/status`, one UTC line per event)

| step | UTC | minutes | exit | what |
|---|---|---|---|---|
| prelude | 01:51 | 0 | 0 | persistence on, 700 W, the registered locks 1410 1500 1605 (`lock-plan.txt`) |
| bytes | 01:51 | 78 | 0 | byte pages at the 1710 lock, G = 1 2 4 16 8 32 3 64, treads 1 to 9 (`results/2026-09-28-nvidia_gh200_480gb-r3-counters/lock1710/`); no base-clock control (no 8x22B prediction reads it) |
| calibrate | 03:09 | 0 | 0 | the ruler stands (triad 3726.3 GB/s, `not_throttled` PASS): `../calibration/` |
| timed | 03:09 | 36 | 0 | locked R3 at 1710: G=8 and 32 (treads 6), G=3 (treads 8); every lock held (worst cell 1710.0 MHz); the first G=8 page INVALID on its VALIDITY gate (NATIVE's repeats) |
| floor | 03:45 | 3 | 3 | floor counters G=64 at base (`r3f-g64.json`) and at the 1710 lock (`r3f-g64-lock1710.json`, FL1 FAIL: the lock was not in force, below) |
| deep | 03:48 | 8 | 1 | locked R3 at 1710 to tread 9: G=4 slipped (a cell at 1680 MHz) |
| recovery | 03:58 | 53 | 0 | by hand after the driver: G=8 timed again at 1710 (VALID), then the deep pages on the lock ladder 1710 1605 1500: G=4 slipped at 1710 again (1635 MHz), G=4 and G=2 held at 1605. `recovery/retake_8x22b.sh` is the script exactly as it ran; its lines are in the same ledger |

**The operator line.** At 02:02 the laptop removed the driver's
`bytes-tail-repriced` flag (`OPERATOR (laptop)` in the ledger): the driver's
G=1 re-price threshold was 8x7B's 420 s, an 8x22B page runs about 1.7 times
longer (583 s), and the flag would have dropped the registered G=64 page. The
threshold now scales with the model (commit 1ab33a6).

**The deep slip is the module's power cap.** The 1710 lock slipped only on
the deep G=4 ladder, with `clocks_event_reasons` 0x4 (software power cap) at
315 to 335 W GPU draw under a 700 W limit: the GH200 module's cap, not the
GPU's. The 0x4 rows are in every ladder's `smi.csv` (`locked-r3/*/smi.csv`:
14 to 22% of the rows, samples down to 1455 to 1485 MHz), timed pages
included; those pages' own per-cell clock readings all held 1710.0 MHz, so
there the dips fell outside the timed calls. The lock ladder held 1605 for
the deep pages.

**The floor lock capture ran unlocked.** `r3f-g64-lock1710.json` records
`held: false` and fits 1801 MHz on w1 and 1783 MHz on w2: `nvidia-smi -lgc`
printed success and the clock stayed boosted. Cycle counts do not depend on
the clock, so both floor pages give the same floor (below). Every lock is now
read back before a capture (commit b994aa0).

## What it answered (registered 2026-09-27, scored on these pages)

Scored on the laptop against the registered JSON, cell by cell; every number
is recomputable from the pages here and that file.

- **Time, at the registered 1710 lock** (G=3 `-p5`, G=8 `-p2b`, G=32 `-p2`):
  40 SHARED and PRIVATE cells, rms 1.71%, worst -4.73% (PRIVATE G=3 n=1), none
  over 5% (registered: rms at or under 2%, no cell beyond 5%). HELD. The miss
  was concentrated at tread 1 (-3.9 to -4.7% at G=3 and 8), a partial second
  wave 8x7B's grids never had; see the timing model's TAIL (commit 5474743).
  G=2 and G=4 were measured only at 1605 (the power cap) and are not scored
  against the 1710 registration.
- **The floor is the tile's.** Cycles per CTA k-step, the slope of
  `sm__cycles_elapsed.avg` over CTA k-steps per SM across n = 2, 3, 4, 6: w1
  346.4 (steps 345.8 347.5 346.0) at base and 347.3 on the unlocked capture;
  w2 348.0 and 346.3 (registered: w1 inside 340 to 365). HELD. Shared-memory
  pipe 82% (w1) and 76 to 80% (w2) of peak.
- **The G >= 8 SHARED slope** over treads 2 to 6: 0.9155 ms per tread at G=8,
  0.9141 at G=32 (registered 0.9270, band 0.908 to 0.946). HELD, near the low
  edge.
- **G=3's period-3 ripple** keeps its phase: steps n = 1 to 8 read 0.559 0.815
  0.990 | 0.939 0.808 0.974 | 0.948 against the registered 0.650 0.830 0.996 |
  0.964 0.810 0.991 | 0.974. HELD. **G=2's zig-zag** (at 1605): 0.597 1.201
  0.766 1.219 0.708 1.210 0.780 1.192, the steps n = 2 to 3, 4 to 5 and 6 to 7
  above the next ones. HELD.
- **Bytes** (the eight lock-1710 counter pages, registered MIX view): of the
  240 SHARED and PRIVATE cells the falsifiers name (every w1 cell, every
  PRIVATE cell, w2 SHARED at G <= 16 and n <= 4), 5 miss by more than 5%:
  PRIVATE w1 G=64 n=9 (-5.4%), PRIVATE w2 G=64 n=5 (+5.3%), w2 SHARED G=3 n=3
  (+6.1%), G=4 n=4 (+6.7%), G=8 n=4 (+5.5%). FALSIFIED on those five cells.
  Over sets: w1 (all three arms, 216 cells) 1.02% rms; PRIVATE w2 1.63%; w2
  SHARED and NATIVE at G <= 16, n <= 4, 3.45%. NATIVE's bytes were predicted as
  SHARED's; four NATIVE w2 cells there also miss by 5.4 to 6.7%.
- **Open questions, printed not scored:** w2 SHARED and NATIVE at G <= 16,
  n >= 5, 2.05% rms (the same cells on 8x7B missed by 5 to 16%); at G >= 32
  (OUT-OF-DOMAIN) 14.0% rms, worst -32.9% (G=64 n=8 and 9, under-predicted).
  The law behind that miss is in `scripts/wave_split_bytes.py` (stage 3,
  commit fb440c5), chosen after these pages were seen.

## Files

- `gh200-driver/status`: the driver's ledger, with the OPERATOR and RECOVERY
  lines; `gh200-driver/summary.txt`
- `logs/`: every capture's and run's console
- `locked-r3/<tag>/`: each locked_r3.py ladder's ledger, `smi.csv` and summary
- `lock-plan.txt`, `power-*.txt`, `clocks-*.txt`: the card's state around each
  step
- `census.json`, `census-mixtral-8x22b.json` and their profiles: the
  preflight's census and the model's
- `recovery/retake_8x22b.sh`: the recovery run by hand after the driver

The `card:` lines of locked_r3.py's ledgers, every page's CARD block and
`PREFLIGHT.json` carry the card's UUID, as every published session does.
