# The calibration this session measured, ADOPTED as the committed ruler on 2026-09-22

`measured_nvidia_h200.yaml` here is what `calibrate` wrote on pod 74osfqvrxtewaw
(EUR-IS-4, H200) at 2026-09-21T23:50Z, from tree 81f80b7: triad bandwidth
4378.0 GB/s (the 2026-09-10 file: 4374.3), ridge 151.4 (155.9), dense bf16
663.0 TFLOP/s at 1455 MHz, and the fp8 GEMM's own clock block reading
`clock_drift_ok: false` (1395 -> 1320 MHz, 5.4%, scored by no gate). Every
report.json under ../results/ cites THESE values.

At publication the committed `moe/bench/hardware/measured_nvidia_h200.yaml`
was left as it was, because adopting this ruler failed 37 tests on the laptop
(18 in test_group_m_sweep, 5 in test_bm128_roofline, the anchor rescore pair,
and singles), the same shape as the 2026-09-09 breakage. It was then adopted on
r3-align in three commits ("H200 calibration 2026-09-21: session 4", the
dependents, the anchor pair): the planted worlds that baked the old ridge into
their expected verdicts now plant fractions of whatever ruler the tree ships,
the four-place pins divide by the rate their own run recorded, and the 19 H200
ladder reports were rescored to 151.4. This copy is the session's own record
and is byte-identical to the adopted file.
