# Rental 2 scorers (2026-10-01)

The scripts that score rental 2's four registrations
(`docs/registered/2026-10-01-rental2-{knobs,launch2,const,w1floor}-gh200.{json,txt}`).
Unlike rental 1's, these were written and tested BEFORE any rental-2 page existed, on
synthetic pages only (`tests/test_scoring_rental2.py`), and are committed with the
registrations in one commit. Each reads only the registered JSON under `<repo>` and the
pages under `<tree>`, writes `<name>.score.{json,txt}` under `<out>`, makes no judgment the
registration does not state, and exits 0 whatever the verdicts.

    python scripts/scoring/rental2/score_knobs.py <repo> <tree> <out>

(the same three arguments for each.)

| file | what it is |
|---|---|
| register.py | writes the four registered JSON and `.txt` files from committed files; `--check` recomputes and compares |
| common.py | the band, law and pricing arithmetic both register.py and the scorers use |
| score_knobs.py | part 1: L2r against NL and ST on the knob pages, the controls, the partition cross-check, T5 |
| score_launch.py | part 2: P0 to P8 of the launch-floor rerun, from cells.csv, hostprobe*.csv, manifest.json and the traces |
| launch_mp.py | rental 1's P6 max-plus model (`timeline`, `mp`), copied verbatim; a test pins the copy |
| score_const.py | part 3: K1 to K5 and the identification, Z by intercept |
| score_w1floor.py | part 4: the per-set slopes, theta and delta, the w2 offset, the co-primary per-cell test |
| f4_implied.py | the diagnosis behind docs/FINDINGS.md's F4 correction note: F4's estimator on the model's own cells |

Pages are found by name under `<tree>`: `*-nvidia_gh200_480gb-<model>-<label>-r3-counters/`
for byte pages (`lock1710/r3c-g<G>.json`) and floor files (`r3f-g64[-lock<F>].json`), and
`*-launch-floor-r2/<model>/` for the launch floor. A page that is missing is reported as NOT
SCORED, never guessed.
