# The calibration this session measured on an H100, NOT adopted as a ruler

Every number here is the NVIDIA H100 80GB HBM3's, and no H200 number is set
beside these.

`measured_nvidia_h100_80gb_hbm3.yaml` here is what `calibrate` wrote on the
Lambda VM (1x NVIDIA H100 80GB HBM3, GPU-4e782590-2a78-3dae-2f1b-a3ce1000a28f,
driver 580.105.08, CUDA 13.0, 700 W power limit, 1980 MHz maximum SM clock) at
2026-09-25T13:50:52Z (its `provenance.utc`), in chain session
`alpha_g-nvidia_h100_80gb_hbm3-20260925T134737Z`, from tree f49a213 with
`measured_dirty: false` (`git_dirty_files: 0`).

`nvidia_h100_80gb_hbm3-buffer_gb8.0-ceilingtriad-gemm_n8192-settletrue-settle_s30.0-5da321b5/`
is the run directory calibrate wrote it into, copied whole: its
`measured_nvidia_h100_80gb_hbm3.yaml` is byte-identical to the one beside this
README (sha256 `71f5fa8ad0ed3be2b8cfd5d043b4f4291aae9b86015dc648c6d16d1e30adeb54`),
and its `cells.csv` holds the five bandwidth patterns. On the VM the first file
was `/home/ubuntu/moe/repo/moe/bench/hardware/measured_nvidia_h100_80gb_hbm3.yaml`
and the run directory was under `/home/ubuntu/moe/repo/results/calibration/`.
These are the only 3 files of the tarball's `moe/repo/` copy that are not in git
at f49a213 (`../session/EXFIL-NOT-PUBLISHED.txt`).

calibrate exited DONE in 30 s (`../session/alpha_g-nvidia_h100_80gb_hbm3-20260925T134737Z/ARMS.tsv`),
and its six RESULT lines all read PASS
(`../session/alpha_g-nvidia_h100_80gb_hbm3-20260925T134737Z/logs/calibrate.log`).

| quantity | this calibrate (from the yaml and `calibrate.log`) |
|---|---|
| triad bandwidth (the ceiling) | 3101.5 GB/s |
| read_stream, read_reduce, copy, write | 3238.8, 3187.0, 3055.3, 3293.1 GB/s |
| derived pin rate | 3352.3 GB/s (memory clock 2619 MHz, 5120-bit bus, from nvml) |
| dense bf16 | 694.4 TFLOP/s (GEMM 8192^3), 93.1% of the 746.1 TFLOP/s the silicon can do at the measured clock |
| bf16 GEMM clock samples | 1395 1365 1380 1365 1380 MHz, median 1380, spread 2.15%, `clock_drift_ok: true` |
| fp8 | 1435.9 TFLOP/s, clock samples 1365 1395 1350 1380 1380 MHz, median 1380, `clock_drift_ok: true` |
| `clock_established` | true |
| ridge (triad) | 223.9 |
| ridge band (read_stream to triad) | 214.4 to 223.9 |
| ridge over the five patterns | 210.9 (write) to 227.3 (copy) |
| settle | compute reached at 1380 MHz, memory at 1980 MHz |
| `measured_dirty` | false |

At f49a213 there is no committed ruler for this card: `moe/bench/hardware/`
holds `h200_nvl.yaml`, `h200_sxm.yaml`, `measured_nvidia_a100_sxm4_80gb.yaml` and
`measured_nvidia_h200.yaml`, and `../session/setup_vm.log` prints "roofline profile
NO MEASURED PROFILE, and no datasheet entry either" for this card.

calibrate printed `PUBLISHED to
/home/ubuntu/moe/repo/moe/bench/hardware/measured_nvidia_h100_80gb_hbm3.yaml`, and
the VM's work tree went from 0 to 1 dirty file
(`../session/alpha_g-nvidia_h100_80gb_hbm3-20260925T134737Z/chain-logs/preconditions.log`).
Every R3 `report.json` under `../results/gaps-nvidia_h100_80gb_hbm3/private_weight_reference/`
therefore carries `git_dirty: true` and `git_dirty_files: 1`, and cites THIS
yaml's values: ridge 223.89, bandwidth 3101.5 GB/s, roof 694.39 TFLOP/s,
reference clock 1380 MHz. The counter pages under
`../results/2026-09-25-nvidia_h100_80gb_hbm3-r3-counters/` ran before calibrate:
they and their `summary.json` record `ridge_source` and `bandwidth_source` as
null and `git_dirty: false`.

This ruler is NOT adopted. Nothing under `moe/bench/hardware/` changes on this
branch; the yaml exists in the repository only here, as a record. Whether to
adopt a ruler for this card is the owner's call; the file is kept so that
decision can be made from the measurement rather than from this note.
