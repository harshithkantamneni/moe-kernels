# Real ncu output

`raw_wide_a100_ncu2025.3.1.csv` is the first `ncu --csv --page raw` this repo
has from a box whose counters were readable: Lambda Cloud, 1x
NVIDIA A100-SXM4-40GB (GPU-5b67366f-3d8c-12eb-32df-400838f594e1), driver
580.105.08, `/usr/local/cuda-13.0/bin/ncu` (Nsight Compute 2025.3.1), run as
root through `sudo` on 2026-09-25. Command: the sixteen metrics of the r3-arms
family's STRICT, CROSS-CHECK and RECORDED classes, `--launch-count 2`, over a
two-kernel torch program (fill 2^26 floats, then double them).

It is WIDE: one row per launch, metrics as columns, the second row the units.
The units it prints are `byte`, `sector`, `ns` (not `nsecond`), `%`, `block`,
`register/thread`, and nothing for `launch__grid_size` and
`launch__waves_per_multiprocessor`. The second launch reads 268,440,832 bytes
from DRAM, which is the 2^26 x 4-byte input plus about 5 KB.
