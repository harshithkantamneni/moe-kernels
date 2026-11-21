# H200 session 5: the alpha(G) chain

Pod w226zpjmj8p1d3, 1x H200 SXM in EUR-IS-4, GPU-0ffa33b8-eeac-8768-2331-dc7e2f8d6490,
CUDA 13.0 host, driver 580.159.04, 700 W, 1980 MHz maximum SM clock. Up
2026-09-23 16:28 to 21:58 UTC (5 h 29 min, about $25). Tree 33d2833 (r3-align);
venvs torch 2.13.0+cu130 (base) and vLLM 0.27.1 (vllm). Driven by
`scripts/alpha_g_chain.sh`.

## What is here

- `alpha_g-nvidia_h200-20260923T163248Z/`: the chain's session directory, whole.
  `CHAIN.tsv` is the ledger (one row per step and pass); `ARMS.tsv` is the
  driver's ledger for the preconditions; `DEVICE` and `R3_DUTY` are what the chain
  pinned the session to; `PAIRS.tsv`, `PAIRS-by-G.tsv`, `PAIRS-fixed.tsv` are the
  alpha(G) tables and `PAIRS-README.txt` is their legend; `chain-logs/` holds one
  log per step (plus each arm's `.price.log` and the end suite's `suite.log`);
  `logs/` holds the driver's logs for thermal, calibrate and pin_probe-n64-g1, and
  `counter_plan.log`, a printed DRAM counter plan (`chain-logs/preconditions.log`
  reads `SKIP counter_plan` and `ARMS.tsv` has no row for it);
  `pin-probe-n64-g1/` is pin_probe's output.
  `chain.lock` is kept: it is not empty, it holds `57903 421a523e82cb` (a PID and
  the pod's hostname).
- `alpha_g_chain.out`: the chain's console, its three passes appended in order.
- `alpha_g_dry.out`: the console of the dry run that priced the chain. Its own
  session directory (`alpha_g-nvidia_h200-20260923T162926Z`) was not exfiltrated.
- `../results/gaps-nvidia_h200/`: all 19 run directories the arms wrote under the
  pod's `MOE_RESULTS_DIR`, 12 in `private_weight_reference/` (R3) and 7 in
  `clock_elasticity/` (R1), including the superseded and partial ones below.
- `../calibration/`: the ruler calibrate measured and its run directory; NOT
  adopted, see its README.

Paths inside the ledger and the consoles are the pod's: `/workspace/session/<dir>`
is `./<dir>` here and `/workspace/results/gaps-nvidia_h200` is
`../results/gaps-nvidia_h200`. Times below are UTC, from the ledger and from the
file mtimes the tarball carries.

## What ran, in order

1. **Dry run** (`alpha_g_dry.out`): every arm's own `--dry-run`; the chain priced
   at 259 min, about $19.79 at $4.59/h, "Book 6 h". Elasticity states 1.0 0.7 0.5.
2. **Pass 1** (console header `NEW`), elasticity states 1.0 0.7 0.5.
   - preflight-r1, preflight-r3: DONE.
   - preconditions: DONE. thermal PASS, 1455 MHz median under load, first and
     last third 0.0% apart; calibrate DONE (triad 4378.2 GB/s, ridge 152.9);
     pin_probe-n64-g1 DONE.
   - gpu-tests (`tests/test_gpu.py` on the card): 37 passed, 1 skipped.
   - probe-check: P1 PASS, the alignment probe under a CUDA graph at 5.30 us per
     call against an eager p50 of 24.42 us.
   - R3 seed 0 at G = 1, 4, 16, 64, duty 0.25 (16:38 to 17:26): CLAIM_FAIL each.
   - r1-g1 (`fa4367d5`, 17:26 to 17:40): INVALID on V4, 70 of 312 rows excluded
     for drift (22.4% against a 20% gate); per duty state, per the ledger, 1.0 0%,
     0.7 30.8%, 0.5 36.5%.
   - r1-g4 (`c07cd22c`) started at 17:41 and was killed about 2 min in.
3. **Owner decision 1**, ledger row `r1-g1 SUPERSEDED`: R1 duty states 1.0 0.7 0.5
   become 1.0 0.5 0.25. The chain was killed with r1-g4 running; that run left the
   partial directory `c07cd22c` and no ledger row. The first r1-g1 log was renamed
   `chain-logs/r1-g1.states-1.0_0.7_0.5.log`.
4. **Pass 2** (console header `NAMED`), elasticity states 1.0 0.5 0.25. The gated
   steps before R3 printed nothing; R3 seed 0 latched and skipped.
   - r1-g1 (`0d8858eb`, 17:43 to 18:03): CLAIM_FAIL, eta 0.3970 [0.3873, 0.4066],
     STRADDLES (the band edge at 0.4).
   - r1-g4 (`e1c429b7`, 18:03 to 18:22): CLAIM_FAIL, eta 1.1291 [1.0683, 1.2004],
     CLOCK-CARRIES.
   - r1-g16 (`a5a8fde2`, 18:22 to 18:41): INVALID on V5 alone, 1 of 295 kept rows
     moved more than 0.05 inside its burst; eta 1.0605 [1.0316, 1.1989].
   - r1-g64 (`a3d5cd3a`, 18:41 to 19:00): CLAIM_FAIL, eta 1.1220 [1.0373, 1.2124],
     CLOCK-CARRIES.
   - R3 seeds 1 and 2 at every G (19:01 to 20:36): CLAIM_FAIL each.
   - end suite, 20:36 to 21:34: interrupted, see below.
5. **Owner decision 2**, ledger row `r1-g16 SUPERSEDED`: re-run r1-g16 once. Its
   first directory was renamed
   `...-a5a8fde2.first-v5-invalid` and its first log
   `chain-logs/r1-g16.first-v5-invalid.log`.
6. **Pass 3** (console header `NAMED`): the gated steps before R3 printed nothing;
   every R3 step and r1-g1, r1-g4, r1-g64 latched and skipped.
   - r1-g16 (`a5a8fde2`, 21:36 to 21:55): INVALID on V7, eta 1.1384
     [1.0930, 1.2749], an interval that does not intersect the admissible
     [-0.075, 1.075].
   - suite: SKIPPED, `END_SUITE=skip`.

## The readings

R3 (`private_weight_reference`), mixtral-8x7b, BLOCK_M=32, duty 0.25, 3 seeds per
G. All 12 pages PASS V0 to V8; C1 FAILS on every page (the ratio lies outside the
refit band ALPHA_BAND [0.529, 0.588)); C2 PASSES on every page. From
`PAIRS-by-G.tsv`:

| G | mean ratio | sd | envelope of the 90% intervals |
|---|---|---|---|
| 1 | 0.9150 | 0.0003 | [0.9144, 0.9156] |
| 4 | 0.7056 | 0.0027 | [0.7033, 0.7090] |
| 16 | 0.6803 | 0.0032 | [0.6767, 0.6846] |
| 64 | 0.6170 | 0.0019 | [0.6128, 0.6193] |

R1 (`clock_elasticity`) at states 1.0 0.5 0.25, the directory each eta in the
PAIRS tables matches:

| G | directory | eta [95%] | word | exit |
|---|---|---|---|---|
| 1 | `0d8858eb` | 0.3970 [0.3873, 0.4066] | STRADDLES | CLAIM_FAIL |
| 4 | `e1c429b7` | 1.1291 [1.0683, 1.2004] | CLOCK-CARRIES | CLAIM_FAIL |
| 16 | `a5a8fde2` | 1.1384 [1.0930, 1.2749] | withheld | INVALID (V7) |
| 64 | `a3d5cd3a` | 1.1220 [1.0373, 1.2124] | CLOCK-CARRIES | CLAIM_FAIL |

## Superseded and partial run directories, and why they are kept

| directory (`clock_elasticity/`) | what it is |
|---|---|
| `...duty1.0_0.7_0.5-...-fa4367d5` | r1-g1 at the first states, INVALID on V4: the page owner decision 1 was taken on |
| `...duty1.0_0.7_0.5-...-c07cd22c` | r1-g4 at the first states, killed about 2 min in: `CARD`, `DEVICE`, `triton-cache/` and a 31-row `cells.csv`, no report, no ledger row |
| `...duty1.0_0.5_0.25-...-a5a8fde2.first-v5-invalid` | r1-g16's first page, INVALID on V5 alone: the page owner decision 2 was taken on; renamed because the re-run carries the same run id and writes the same directory name |

Each is the evidence a decision was taken on, and publishing every directory
the arms wrote keeps this tree comparable, file for file, with the exfil
tarball. None of their values is in the PAIRS tables: the R1 etas there are
those of the directories in the R1 table above.

## The end suite, interrupted

`chain-logs/suite.log`: the whole suite, uncapped, from the base venv. It ran from
20:36 to 21:34 (3478 s) and was interrupted by the owner at 49%
(`KeyboardInterrupt`): 10 failed, 2470 passed, 19 skipped in 57:51. The ledger
row reads `suite ERROR 2` (pytest exit 2). No data was touched: the last file under
the results root before it was written at 20:36:41 (R3 `b150346d`, G=64 seed 2)
and the next at 21:36:18 (the r1-g16 re-run).

The 10 failures, as the log names them:

- four name `2026-09-15-nvidia_h200-session3`, a directory that was present in
  the pod's `results/published/` and that 33d2833 does not track (it is tracked
  on `pod-h200-session3`): `test_calibration_provenance.py` x3
  (`test_every_published_arm_has_a_declared_verdict`,
  `test_only_whole_layer_is_blocked_among_arms_that_carry_rows`,
  `test_the_committed_report_is_what_the_code_produces_today`) and
  `test_docs.py::test_readme_arm_counts_are_the_trees` (15 arms against the
  README's 14);
- `test_block_m_crossing_sweep.py::test_every_report_carries_a_provenance_block_with_the_audited_keys`
  ("there is no card on this box") and
  `test_blockk_diagonal.py::test_an_absent_card_may_be_named_and_a_present_one_may_not_be_contradicted`
  ("this test box has a CUDA device; the laptop path is what is asserted here");
- `test_bm128_roofline.py` x3 (`test_the_self_test_passes_every_one_of_its_own_gates`,
  `test_a_working_self_test_exits_done`,
  `test_both_directions_of_the_uncontrolled_mode_are_planted`) and
  `test_bn_decomposition.py::test_the_four_height_self_test_still_separates_its_planted_worlds`:
  each self-test exits 3 (`INVALID`). The failing gate of each, as the log
  prints it:
  - `bm128_roofline --self-test`, "11 PASS, 1 FAIL, 0 UNKNOWN": gate
    `S_hypothesis_roof_refused`, "a run on a roof no attached device measured
    reaches no verdict", gate "every world's real verdict is 'NOT SETTLED'",
    saw real verdicts in all five worlds (suite.log line 422, in the captured
    stdout of `test_a_working_self_test_exits_done`; the other two tests print
    the summary line or `assert 3 == 0` only).
  - `bn_decomposition --self-test`, "4 PASS, 1 FAIL, 0 UNKNOWN": gate `S4`, "the
    design resolves alpha_a", gate `sd(alpha_a) <= 0.025` in the TRUTH world,
    saw `sd = 0.0440 at GROUP_SIZE_M=16, 17 reps` (suite.log line 694).

  At 33d2833, on a laptop with no CUDA device (torch 2.13.0 CPU), with
  `../calibration/measured_nvidia_h200.yaml` copied over the committed
  `moe/bench/hardware/measured_nvidia_h200.yaml`, the same four tests pass (4
  passed).

## Provenance of these files

The exfil tarball `exfil-session5-alpha_g.tar.gz`, sha256
`34032d857e62ff7abc01152a46c19022b16b044be7fa38519f668c7c48e15454`, holds 464
files. Every one was copied here and compared by sha256: 464 identical, 0
different. 19 of them are not in git: Triton's
`cuda_utils.cpython-312-x86_64-linux-gnu.so`, one per run directory under
`triton-cache/MIH4X24CEAJDWYGXCZ4EKW6DTWSUG2ZBSBPV7NGI62HAUKJ7F7BQ/`, which the
repository's `.gitignore` drops by its `*.so` rule (its `results/published/`
block re-includes `.ptx`, `.cubin`, `.ncu-rep`, `.nsys-rep` and `.qdrep`, not
`.so`). Session 4's published tree carries no `.so` and no such directory
either. The other 445 are tracked, beside this README, `../calibration/README.md`
and `../KIND`.
