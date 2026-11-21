# The calibration this session measured, NOT adopted as the committed ruler

`measured_nvidia_h200.yaml` here is what `calibrate` wrote on pod w226zpjmj8p1d3
(1x H200, GPU-0ffa33b8-eeac-8768-2331-dc7e2f8d6490, driver 580.159.04, CUDA 13.0,
700 W power limit, 1980 MHz maximum SM clock) at 2026-09-24T04:29:01Z, in chain
session `alpha_g-nvidia_h200-20260924T042526Z`, from tree c09b353 with
`measured_dirty: true` (`git_dirty_files: 1`).

`nvidia_h200-buffer_gb8.0-ceilingtriad-gemm_n8192-settletrue-settle_s30.0-2ea5b191/`
is the run directory calibrate wrote it into, copied whole: its
`measured_nvidia_h200.yaml` is byte-identical to the one beside this README
(sha256 `98c7da226a3d62ac6d693eb34cc857fbb8ce4764fea9e19346384d5cc6beb9af`),
and its `cells.csv` holds the five bandwidth patterns. calibrate's six RESULT
lines all read PASS
(`../session/alpha_g-nvidia_h200-20260924T042526Z/logs/calibrate.log`).

The same run directory was first written by the first chain session's calibrate
(`alpha_g-nvidia_h200-20260924T042015Z`, log closed at 04:23:49), which exited
INVALID on `clock_established` and had printed `PUBLISHED to` the tracked
`moe/bench/hardware/measured_nvidia_h200.yaml` before that gate was scored. The
second calibrate carries the same run id and overwrote both. Neither the first
calibrate's yaml nor its `cells.csv` is in the tarball; its numbers are those its
log prints (`../session/alpha_g-nvidia_h200-20260924T042015Z/logs/calibrate.log`),
in the second column below.

| quantity | this session (`042526Z`, DONE) | first calibrate (`042015Z`, INVALID; log only) | session 5, same card, 2026-09-23 | committed ruler at c09b353 (session 4, 2026-09-21) |
|---|---|---|---|---|
| triad bandwidth | 4377.0 GB/s | 4376.9 GB/s | 4378.2 GB/s | 4378.0 GB/s |
| read_stream bandwidth | 4613.2 GB/s | 4612.8 GB/s | 4613.0 GB/s | 4612.9 GB/s |
| dense bf16 | 649.9 TFLOP/s, GEMM clock median 1455 MHz | 657.3 TFLOP/s, 1455 MHz | 669.6 TFLOP/s, 1470 MHz | 663.0 TFLOP/s, 1455 MHz |
| bf16 GEMM clock samples | 1440 1455 1440 1470 1455, spread 2.0% | 1455 1455 1485 1545 1455, spread 5.8% | 1470 1440 1470 1440 1470, spread 2.0% | 1455 1440 1455 1455 1470, spread 2.0% |
| `clock_established` | true | false | true | true |
| ridge (triad) | 148.5 | 150.2 | 152.9 | 151.4 |
| ridge band (triad to the matched `read_stream` ceiling) | 140.9 to 148.5 | 142.5 to 150.2 | 145.2 to 152.9 | 143.7 to 151.4 |
| ridge over the five patterns (`write` lowest, `copy` highest) | 138.8 to 151.1 | 140 to 153 (the log prints whole numbers) | 142.9 to 155.7 | 141.6 to 154.1 |
| fp8 | 1446.1 TFLOP/s, clock median 1365 MHz, `clock_drift_ok: true` | clock median 1395 MHz, DRIFT PASS; the log prints no fp8 rate | 1424.3 TFLOP/s, 1380 MHz, `clock_drift_ok: true` | 1437.0 TFLOP/s, 1395 MHz, `clock_drift_ok: false` |
| `measured_dirty` | true | not in the log | false | false |

Against the committed ruler, this session's triad bandwidth is 1.0 GB/s lower,
its dense bf16 2.0% lower, and its ridge 1.9% lower.

Every R3 `report.json` under
`../results/gaps-nvidia_h200/private_weight_reference/` cites THIS yaml's
values: ridge 148.49, bandwidth 4376.99 GB/s, roof 649.92 TFLOP/s, reference
clock 1455 MHz. The R1 report under `../results/gaps-nvidia_h200/clock_elasticity/`
records no ridge and no bandwidth (its provenance block reads "not supplied by
caller").

At publication the committed `moe/bench/hardware/measured_nvidia_h200.yaml` was
left as it was at c09b353 (ridge 151.4), which is also the file at r3-align
60db8ba. Whether to adopt this ruler, and to fix whatever depends on the old one,
is the owner's call; the file is kept so that decision can be made from the
measurement rather than from this note.
