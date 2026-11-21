# Calibration provenance: 2026-09-01-nvidia_h200-alpha-0558

Verdict: **same_session**

- `arm`: 2026-09-01-nvidia_h200-alpha-0558
- `calibration`: measured.yaml
- `yaml_gpu`: NVIDIA H200
- `yaml_commit`: d9190b276b8c9fe4c5caca14b5da95dfe49ab817
- `yaml_checked_on`: 2026-09-01
- `yaml_bandwidth_gbps`: 4373.8838993043155
- `yaml_peak_tflops`: {'bf16': 715.999417539747, 'fp16': 715.999417539747, 'fp8_e4m3': 1443.4135858041589, 'fp8_e5m2': 1443.4135858041589}
- `rows`: 3696
- `timed_rows`: 3696
- `row_commits`: ['d9190b276b8c9fe4c5caca14b5da95dfe49ab817']
- `row_span`: ('2026-09-01T22:27:45', '2026-09-01T23:19:05')
- `row_gpus`: ['NVIDIA H200']
- `row_dtypes`: ['bf16']
- `row_bandwidth_gbps`: [4373.8838993043155]
- `row_peak_tflops`: [('bf16', 715.999417539747)]
- `ceilings`: agree
- `ceiling_disagreements`: []
- `device`: agrees
- `commit`: matches
- `date`: within the sweep

Ridge: 2026-09-01-nvidia_h200-alpha-0558: 163.7 FLOP/byte for bf16 from the ceilings stamped on the rows (same_session)
