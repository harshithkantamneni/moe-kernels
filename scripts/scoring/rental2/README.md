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
| common.py | the band, law and pricing arithmetic both register.py and the scorers use; the gate addendum's views (`View`, `two_views`, `combine_verdict`) |
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

## Gate-failed pages (the 2026-10-02 addendum)

`docs/registered/2026-10-02-rental2-addendum-gates.{json,txt}`, fixed after the rental-2
pages' gate results were seen and before any score; the four registrations are unchanged
(their sha256 is pinned in the addendum and by a test). Every scorer's `score()` runs its
`score_view(repo, tree, view)` twice through `common.two_views`:

- **ALL** counts every page; **CLEAN** drops every page with a VALIDITY gate other than
  PASS (FL1 on a floor capture; V0 to V10 on a byte page; CLAIM gates C1 to C6 gate
  nothing). A floor capture under a lock is judged in CLEAN by the null kernel's lock
  check (its median `sm_clock_mhz` within 3% of the lock) in place of FL1, by FL1 when it
  has no null kernel; FL1 is still printed. A page whose V1 (launch attribution) failed is
  unusable in both.
- In both, a floor capture taken under a lock loses every (cell, GEMM) whose own
  `sm_clock_mhz` is over the lock by more than 15 MHz (one step), and each is recorded.
  Every clock a scorer reads is the cells' own `sm_clock_mhz`, never the lock asked, and
  it enters a verdict only as a ratio (K3's f_1005 / f_1710), since it reads about 1.4% low
  uniformly; K1, K2, K4 and part 4 are in SM cycles, part 1 in bytes, part 2 in milliseconds
  against the registered C_reg, none reading a clock.
- The registered verdict is the common one when ALL and CLEAN agree, `INCONCLUSIVE (ALL
  x; CLEAN y)` when they differ, and `NOT SCORED (CLEAN has no data); ALL, reading only
  gate-failed pages: x` when only ALL has data.
- Part 1 keeps its registered V6 and V10 handling (`per_part.knobs.gates_not_gating`):
  V6 flags the page in both views, V10 gates nothing. Part 2's directories record no
  gate, so its two views agree by construction.

`<name>.score.json` holds the registered tree (ALL's numbers, the combined verdicts) and
`addendum`: the page record (gates failed, cells dropped, what each view did), every
verdict path with its ALL, CLEAN and registered reading, and the CLEAN tree. The `.txt`
ends with the pages a gate or the clock rule touched and every verdict that differs.

## Scored 2026-10-02

Run unchanged on `results/published/2026-10-02-nvidia_gh200_480gb-rental2-session`; the
outputs are `{knobs,launch,const,w1floor}.score.{json,txt}` in this directory and
`SCORES.md` tabulates every registered prediction (ALL, CLEAN, registered verdict, key
numbers). No scorer crashed or failed to read a page, and none was edited after the pages:
there is no post-page fix. `SCORES.md` ends with two readings left to the owner.
