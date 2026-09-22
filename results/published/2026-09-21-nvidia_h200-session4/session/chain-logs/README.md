# The three chains this session ran on the pod, and how each ended

- `session4-chain.sh` / `.log` -- chain 1: R3 at G=16, then the driver over R1 and
  R2 (`session4-r1r2.log`), then the suite capped at 50 failures
  (`session4-pytest.log`: 50 failed / 2185 passed / 7 skipped in 24 min).
- `session4-chain2.sh` / `.log` -- chain 2: the clock-lock probe
  (`session4-clocklock.log`: REFUSED, "The current user does not have
  permission to change clocks"), the G=1 `--seed 1` replicate, qwen2 at G=1,
  and `counter_plan` (`session4-counter.log`: REFUSED, no `ncu` on the image).
- `session4-chain3.sh` / `.log` -- chain 3: the whole suite UNCAPPED
  (`session4-pytest-full.log`). TERMINATED BY HAND at ~51% after 1 h 33 min and
  carries NO summary. It was not hung: it was inside tests/test_h200_gaps_session.py,
  whose tests spawn the driver's own `--dry-run`, and on a box with a card and vLLM
  present every arm's dry run really plans (~55 s each) where the laptop's refuse
  at once. Hundreds of those tests is hours of card for a list whose shape the
  capped run already gave: (a) tests asserting OFF-GPU refusals that do not skip
  on a GPU box (test_group_m_sweep x10, clock_elasticity, dtype_tile_confound,
  driver); (b) calibration-derived literals broken by this session's yaml
  (`0.6443`, `152.38 < ridge`); (c) GPU-only staleness --
  `test_full_calibration_runs` pins 4 bandwidth patterns, and
  `test_graph_policy_skips_a_long_kernel` drives graph mode with a hardware
  profile that carries no reference clock.
