# The calibration this box measured, NOT adopted as a committed ruler

Every number here is the NVIDIA GH200 480GB's (nvidia_gh200_480gb). None of it
is an H200 number.

`measured_nvidia_gh200_480gb.yaml` here is what `calibrate` wrote on the GH200
(GPU-33948cb3-46ff-1834-07b0-6d05730ad309, driver 580.105.08, CUDA 13.0, 1980
MHz maximum SM clock), in chain session
`alpha_g-nvidia_gh200_480gb-20260925T071107Z`, from tree f49a213. Its
provenance block reads `utc: '2026-09-25T07:14:12+00:00'`, `git_dirty: false`,
`git_dirty_files: 0`, and its top level `measured_dirty: false`.

`nvidia_gh200_480gb-buffer_gb8.0-ceilingtriad-gemm_n8192-settletrue-settle_s30.0-28dc063d/`
is the run directory calibrate wrote it into, copied whole: its
`measured_nvidia_gh200_480gb.yaml` is byte-identical to the one beside this
README (sha256 `242b62b8405fe65cc3a048481e59a8470bd6db0e26f4705e637cf52ffe17e3bc`),
and its `cells.csv` holds the five bandwidth patterns. On the VM the run
directory sat under the checkout, at
`/home/ubuntu/moe/repo/results/calibration/<dir>`, and the yaml was also
written to `/home/ubuntu/moe/repo/moe/bench/hardware/measured_nvidia_gh200_480gb.yaml`;
both paths are in the tarball under `moe/repo/`.

calibrate's six RESULT lines all read PASS, and its log prints `PUBLISHED to
/home/ubuntu/moe/repo/moe/bench/hardware/measured_nvidia_gh200_480gb.yaml`
(`../session/alpha_g-nvidia_gh200_480gb-20260925T071107Z/logs/calibrate.log`).
The chain's `chain-logs/preconditions.log` records the work tree at 0 dirty
files after thermal and 1 after calibrate ("this arm dirtied the work tree (0
-> 1 files)"). That one file is why every R3 `report.json` measured after it
carries `git_dirty: true` and `git_dirty_files: 1`. The five counter pages and
their `summary.json` were written before calibrate ran and carry `git_dirty:
false`.

## What it measured

From the yaml and `calibrate.log`:

| quantity | value |
|---|---|
| triad bandwidth (the ceiling) | 3725.1 GB/s |
| read_stream / read_reduce / copy / write | 3885.2 / 3818.8 / 3667.7 / 3950.4 GB/s |
| pin rate | 4022.8 GB/s, from a 6144-bit bus read through NVML at a 2619 MHz memory clock (the table's 6016 bits would give 3939.0 GB/s) |
| dense bf16 | 662.8 TFLOP/s (8192^3 GEMM), GEMM clock median 1455 MHz, samples 1455 x 5, spread 0.0%, DRIFT PASS |
| fp8 | 1528.5 TFLOP/s, GEMM clock median 1350 MHz, samples 1365 1365 1350 1350 1350, spread 1.1%, `clock_drift_ok: true` |
| ridge (triad) | 177.9 FLOP/byte |
| ridge band (`ridge_band`: read_stream, the pattern matched to this study's traffic, to triad, the ceiling) | 170.6 to 177.9. The five patterns alone span 167.8 to 180.7: write 167.8, read_stream 170.6, read_reduce 173.6, triad 177.9, copy 180.7 |
| settle | compute reached at 1440 MHz, memory at 1980 MHz |
| `clock_established` | true |
| `observed.power_limit_w` | 900.0 |

## Who cites it

Every R3 `report.json` under
`../results/gaps-nvidia_gh200_480gb/private_weight_reference/` (the chain's
three pages at the unlocked clock and the five 1710 MHz pages, all at duty
0.25) cites THIS yaml's values: ridge
177.93, bandwidth 3725.12 GB/s, roof 662.82 TFLOP/s, reference clock 1455 MHz.
The two partial run directories there (`...-257313aa`, `...-32bec6b2`) have no
report. The counter pages under `../results/2026-09-25-nvidia_gh200_480gb-r3-counters/`
and their `summary.json` record `ridge_source` and `bandwidth_source` as null;
the thermal report under `../results/gaps-nvidia_gh200_480gb/thermal_acceptance/`
records no ridge.

## Not adopted

At f49a213 `moe/bench/hardware/` holds four yaml files (`h200_nvl.yaml`,
`h200_sxm.yaml`, `measured_nvidia_a100_sxm4_80gb.yaml`,
`measured_nvidia_h200.yaml`) and the `vllm_configs/` directory, and no GH200
file; `../session/setup_vm.log` prints `roofline profile NO MEASURED
PROFILE, and no datasheet entry either`. This publication adds nothing under
`moe/bench/hardware/` and changes nothing there. Whether to adopt this ruler
for nvidia_gh200_480gb is the owner's call; the file is kept so that decision
can be made from the measurement rather than from this note.
