THE alpha(G) TABLES, rebuilt from the reports on disk by
scripts/alpha_g_chain_helpers.py pairs-table at the end of every chain pass and
before every STOP, and again on the laptop after exfil by the same command.

PAIRS.tsv, one row per ratio run (private_weight_reference), G then seed:
  G seed duty run_id   the swizzle the run pinned (GROUP_SIZE_M), its seed, its
                       duty and its run id.
  ratio lo hi          the run's OWN reading, slope(shared) / slope(private), and
                       its 90% percentile bootstrap over repeats WITHIN the run
                       (R3's INTERVAL_PCT). It understates the run-to-run spread.
  exit                 the run's own exit word: classify over its page's gates.
  exit_scope           how C1 inside that word was scored: `alone` on this run's
                       interval; `envelope` on the envelope of this run's interval
                       and those of the earlier seeds it was given through
                       --replicate-of. Seed 0's exit and seed 1's can differ by
                       scope, not by result.
  rep_n rep_spread rep_sd env_lo env_hi joint
                       the joint reading on that run's page, over itself and those
                       earlier seeds: n, the spread and sd of the points (an sd
                       needs three), the envelope, and `joint`, C1's verdict on the
                       envelope against R3's refit band ALPHA_BAND [0.529, 0.588). `joint`
                       is NOT a quotability flag: NO-REUSE, expected at G=1, reads
                       FAIL. `none` on a run scored alone, and on a run whose page
                       formed no ratio of its own (its page's replicates block is
                       then the other runs' reading, not one this run is in).
  clk_<arm> low_cells  each arm's median under-load clock (MHz) over the ladder, and
                       the (arm, tread) cells whose level record reads LEVEL LOW.
  eta eta_lo eta_hi band eta_exit
                       R1 (clock_elasticity) at this G: the per-M-tile elasticity
                       over treads 2 and deeper, its 95% percentile bootstrap
                       (clock_elasticity.fit's 2.5th and 97.5th percentiles), the
                       regime word off that interval, and R1's own exit word. The
                       word is withheld (`withheld:<EXIT>`) from a page whose gates
                       did not stand behind it. It is a SECANT between R1's capped
                       duty states, not a reading at R3's duty.

PAIRS-by-G.tsv, one row per G, THE PER-G VALUE: every seed of that G whose report
formed a ratio, whatever order the seeds ran in, read together by R3's own
cross-run machinery (what `--read RUN --replicate-of ...` prints):
  n seeds              how many runs, and their seeds.
  mean sd              the points' mean, and their sd (three points or more).
  env_lo env_hi joint  the envelope of the runs' 90% intervals and C1's verdict on
                       it, the same rule and band as `joint` above.
  invalid_in_envelope  the seeds inside the envelope whose own page exited INVALID:
                       R3 admits them by design, their spread is information and
                       their ratio is not quotable alone.
  eta .. eta_exit      R1 at this G, as above.
  note                 why a G has no joint (no report formed a ratio, or
                       load_replicates refused the set, e.g. two duties).

PAIRS-fixed.tsv: the coordinates every row shares (model, tile, pinned config,
treads, repeats, duty), their value and where each was read; MIXED when the
reports disagree.
