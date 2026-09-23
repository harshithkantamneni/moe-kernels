# Session 4's clock-elasticity cells at G=16, as a test fixture

`cells.csv` here is the R1 (`scripts/clock_elasticity.py`) run of the H200
session of 2026-09-21, trimmed by columns only. `tests/test_clock_elasticity.py`
re-scores it through `read_rows`, `fit` and `gates_for` and pins the verdicts,
so the tread set the claim is fitted over is decided against the cells that
motivated it and not against a planted world.

Source, on branch `pod-h200-session4` at commit
`2c19a4ce7617a6aeaa025614cd97e45a24024723` ("H200 session 4: R3 on a real
card, the elasticity it needed, and two smokes"):

    results/published/2026-09-21-nvidia_h200-session4/results/gaps-nvidia_h200/clock_elasticity/nvidia_h200-burstms40.0-dtypebf16-duty1.0_0.5_0.25_0.1-l2flushtrue-modelmixtral_8x7b-repeats13-s-9f91fa91/cells.csv

Measured by the code at `81f80b7` (the rows' `prov_git_sha`) on one NVIDIA
H200, card stamp `NVIDIA H200`, with `--treads 8 --duty 1.0 0.5 0.25 0.1
--repeats 13 --burst-ms 40 --target-ms 200 --trials 3 --warm-ms 200
--settle-seconds 10` and the default pin, GROUP_SIZE_M=16.

What was trimmed, and nothing else was:

- `instrument`: the same string on all 416 rows, equal to the module's
  `INSTRUMENT`, which is the field's default when the column is absent.
- `detail`: free-text notes (LEVEL HIGH records); no gate and no fit reads it.
- every `prov_*` column: the run's provenance, recorded above instead.

All 416 rows are kept, in the published order, and every other column is the
published text byte for byte. `read_rows` reads by header name, so the absent
columns come back as the dataclass defaults.
