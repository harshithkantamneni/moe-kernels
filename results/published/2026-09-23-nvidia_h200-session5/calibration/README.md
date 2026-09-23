# The calibration this session measured, NOT adopted as the committed ruler

`measured_nvidia_h200.yaml` here is what `calibrate` wrote on pod w226zpjmj8p1d3
(EUR-IS-4, 1x H200 SXM, GPU-0ffa33b8-eeac-8768-2331-dc7e2f8d6490, driver
580.159.04, CUDA 13.0, 700 W power limit, 1980 MHz maximum SM clock) at
2026-09-23T16:36:20Z, from tree 33d2833 with `measured_dirty: false`.

`nvidia_h200-buffer_gb8.0-ceilingtriad-gemm_n8192-settletrue-settle_s30.0-2ea5b191/`
is the run directory calibrate wrote it into, copied whole: its
`measured_nvidia_h200.yaml` is byte-identical to the one beside this README, and
its `cells.csv` holds the five bandwidth patterns. calibrate's six RESULT lines
all read PASS (`../session/alpha_g-nvidia_h200-20260923T163248Z/logs/calibrate.log`).

| quantity | this session | committed ruler at 33d2833 (session 4, 2026-09-21) |
|---|---|---|
| triad bandwidth | 4378.2 GB/s | 4378.0 GB/s |
| dense bf16 | 669.6 TFLOP/s, GEMM clock median 1470 MHz | 663.0 TFLOP/s, 1455 MHz |
| ridge (triad) | 152.9 | 151.4 |
| ridge band over the five patterns | 145.2 to 152.9 | 143.7 to 151.4 |
| fp8 | 1424.3 TFLOP/s, clock median 1380 MHz, `clock_drift_ok: true` | 1437.0 TFLOP/s, 1395 MHz, `clock_drift_ok: false` |

The two triad bandwidths are 0.12 GB/s apart; the ridge moved by 1.0% because the
bf16 GEMM's achieved rate did.

Every R3 `report.json` under
`../results/gaps-nvidia_h200/private_weight_reference/` cites THESE values: ridge
152.94, bandwidth 4378.15 GB/s, roof 669.60 TFLOP/s, reference clock 1470 MHz.
The R1 reports under `../results/gaps-nvidia_h200/clock_elasticity/` record no
ridge and no bandwidth (their provenance block reads "not supplied by caller").

At publication the committed `moe/bench/hardware/measured_nvidia_h200.yaml` was
left as it was at 33d2833 (ridge 151.4). Whether to adopt this ruler, and to fix
whatever depends on the old one, is the owner's call; the file is kept so that
decision can be made from the measurement rather than from this note.
