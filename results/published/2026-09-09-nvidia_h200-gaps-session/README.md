# H200 session, 2026-09-09

The raw session the driver ran, kept whole so every verdict it printed can be
re-derived rather than taken on trust. This is NOT a published arm: it carries
no `measured.yaml` and no `merged.csv` of its own, and the `KIND` file beside
this one says `session` so the calibration-provenance census names it instead
of reporting a missing calibration.

    session/ARMS.tsv        the ledger: one row per arm, its state and its log
    session/logs/<arm>.log  each arm's full stdout, including its RESULT lines
    results/<script>/<run>/ what each arm wrote: cells.csv, report.json, plan.json

The calibration this session measured and published is the committed
`moe/bench/hardware/measured_nvidia_h200.yaml` at ab61e55, not a copy here.

Sixteen arms ran in 33 minutes: five DONE, two REFUSED by design, two
CLAIM_FAIL by design, and six INVALID. What each verdict means, and which of
the six were the clock rule rather than the measurement, is in `docs/FINDINGS.md`.
