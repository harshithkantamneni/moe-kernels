# 2026-08-26-nvidia_h200-full-three-way-recalibrated

Derived from `2026-08-26-nvidia_h200-full-three-way` by `scripts/recompute_ceilings.py`.

The measurements are identical. `ms_p50`, `tflops`, `compulsory_gbps` and
`arith_intensity_compulsory` come from the timing and the byte model and
were never affected by the calibration. Only these four differ:

  - `achieved_bw_gbps`
  - `bw_ceiling_pattern`
  - `achieved_peak_tflops`
  - `pct_of_achieved_tflops`
  - `implied_traffic_ratio`

Re-derived against **NVIDIA H200 (measured)**, 4377.0 GB/s (pattern `triad`), where the original arm
used 4374.7 GB/s.

No FINDINGS.md here on purpose. An earlier version of the recompute script
copied it across, which put prose quoting the OLD ceiling directly beside
rows carrying the new one. The analysis in the original arm was written
against the ruler of that day: read it there, and treat these rows as the
corrected numbers. The two arms together are the evidence that the ruler
moved.
