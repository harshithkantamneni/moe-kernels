THE alpha(G) TABLES, rebuilt from the reports on disk by
scripts/alpha_g_chain_helpers.py pairs-table at the end of every chain pass and
before every STOP, and again on the laptop after exfil by the same command.

PAIRS.tsv, one row per ratio run (private_weight_reference), G then seed:
  G seed duty run_id   the swizzle the run pinned (GROUP_SIZE_M), its seed, its
                       duty and its run id.
  ratio lo hi          the run's OWN reading, slope(shared) / slope(private), and
                       its 90% percentile bootstrap over repeats WITHIN the run
                       (R3's INTERVAL_PCT). It understates the run-to-run spread.
                       Both slopes over the window the report records
                       (claim_min_tread): treads 2 and deeper
                       since 2026-09-23, every tread on a report before it.
                       Reports of two windows are refused, not pooled.
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
                       did not stand behind it. It is a SECANT across R1's duty
                       states, not a local reading at R3's duty. This session's R1
                       states: 1.0 0.5 0.25; R3's duty: 0.25 (PAIRS-fixed.tsv's
                       r1_duty and r3_duty). At the chain's defaults, R1_DUTY
                       1.0 0.5 0.25 and R3_DUTY 0.25, the secant runs from the
                       capped clock at 1.0 to the clocks at 0.5 and 0.25 (on
                       session 5's H200 only 1.0 held the 700 W cap), and R3's
                       0.25 is the top of that range.
  reads_as             what the ratio can be read as, off R1's word at this G and
                       the rule H200 session 5's findings support: `re-read fraction`
                       ONLY when the band is RAW-STANDS on a page whose gates
                       stood behind it; `blend (traffic and a clock-scaled on-chip floor): not alpha` on CLOCK-CARRIES (session 5 at
                       G >= 4: the shared arm sits on a per-tile floor that
                       scales with the SM clock, any alpha in [0, 0.60] fits it
                       equally, and the bytes-rate bound below still proves real
                       reuse); `unresolved` on STRADDLES, UNREGISTERED-GAP,
                       withheld:<EXIT> and unmeasured. It is R1's reading at the
                       G; whether this run's ratio is quotable at all is `exit`.

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
  reads_as             as above.
  top_tread shared_top_ms bound_read_stream bound_pin_rate
                       THE BYTES-RATE BOUND, the one bound on alpha that needs no
                       private arm: alpha <= (t x C / W - 1) / (n - 1), with t the
                       shared arm's time at its top tread n off each report's own
                       ladder (shared_top_ms, the mean over these runs), W the
                       expert set (PAIRS-fixed.tsv's expert_set_bytes) and C a
                       ceiling off the ruler the session measured
                       (PAIRS-fixed.tsv's read_ceiling_gbps and pin_rate_gbps):
                       the shared arm reads W once and alpha x W for each later
                       M-tile, and no faster than C. At read_stream, the ruler's
                       read ceiling, and at the pin rate, the bus's hard one. 1 or
                       above excludes nothing (a full re-read fits in the time);
                       `none` when the ruler or the ladder is missing, and
                       PAIRS-fixed.tsv's ruler row says why.
  note                 why a G has no joint (no report formed a ratio, or
                       load_replicates refused the set, e.g. two duties).

PAIRS-fixed.tsv: the coordinates every row shares (model, tile, pinned config,
treads, repeats, duty), their value and where each was read; MIXED when the
reports disagree. Also the bound's inputs: expert_set_bytes (each report's
memory_plan.per_copy_bytes), the ruler (which yaml, and the check that its named
bandwidth is the one the reports were scored against), read_ceiling_gbps and
pin_rate_gbps. On the laptop the session's calibrate yaml is not on disk, so the
tracked ruler is read and checked; `ruler=<yaml>` on pairs-table names another.
