# The calibration this session measured, NOT adopted as the committed ruler

`measured_nvidia_h200.yaml` here is what `calibrate` wrote on pod 74osfqvrxtewaw
(EUR-IS-4, H200) at 2026-09-21T23:50Z, from tree 81f80b7: triad bandwidth
4377.3 GB/s (committed: 4374.2), ridge 151.4 (committed: 155.9), and one entry
with `clock_drift_ok: false`. Every report.json under ../results/ cites THESE
values; the committed `moe/bench/hardware/measured_nvidia_h200.yaml` was left as
it was because adopting this ruler fails 37 tests on the laptop (18 in
test_group_m_sweep, 5 in test_bm128_roofline, the anchor rescore pair, and
singles) -- the same shape as the 2026-09-09 breakage. Whether to adopt it and
fix those dependents is the owner's call; the file is kept so that decision can
be made from the measurement rather than from this note.
