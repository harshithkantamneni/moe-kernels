# Pod runbook

**The next session is the alpha(G) chain**, `scripts/alpha_g_chain.sh`, one
sequenced ledger that runs the driver's preconditions inside its own session
directory; its section, "The alpha(G) chain" below, is the command to book.
Outside that section the driver's commands and the hand-run arms are the
standalone path, for a rental that books the driver or one arm on its own.

Besides the chain, two session scripts exist and this file covers both, in
the order they matter. **Part A** is `scripts/h200_gaps_session.sh`, the
driver: every open experiment as one unattended, resumable pod run with a
ledger. **Part B** is the human-readable companion to
`scripts/pod_session.sh`, the 2026-09-01 sweep session; its pre-flight, its
step gates and its failure playbook still apply to the sweep, and it is kept
here as the operator's page for that script. `docs/RUNPOD.md` is how to get a
pod and an environment; `docs/APPARATUS.md` is what every exit code and every
`RESULT:` line means. Nothing on this page terminates a pod, commits or pushes.

---

# Part A: the session driver, `scripts/h200_gaps_session.sh`

**Before renting anything, on the laptop, free:**

```bash
bash scripts/h200_gaps_session.sh --list       # the arms, in read order, and what each closes
bash scripts/h200_gaps_session.sh --dry-run    # every arm's OWN plan, its cost, the session MDE
```

`--dry-run` runs each arm's own `--dry-run`, records its exit code, and gives
each a word: `PLANNED` (it printed a plan), `PLAN_REFUSED` (it refused before
printing one, which off a GPU box is the refusal working), `BROKEN` (a
traceback, or any code the table does not name; the session exits 3 and the
plan on the page is not a plan). It prints the cost table with every figure
read off the arm's own plan and the command that produced it, the TOTAL in two
forms, and the two `--only` lines a two-hour and a three-hour rental can
actually reach.

**On the pod:**

```bash
cd /workspace/repo
bash scripts/h200_gaps_session.sh                       # every arm
bash scripts/h200_gaps_session.sh --only thermal,calibrate,bn_g16   # a subset; thermal and calibrate belong in EVERY subset
```

## Where things land, and how to resume

- `SESSION=` names the session directory. Default on a pod:
  `/workspace/session/gaps-<card>-<UTC stamp>`; off a pod,
  `results/h200_gaps/session-<card>-<UTC stamp>` under the checkout. The
  ledger is `$SESSION/ARMS.tsv` (`ARMS-dryrun.tsv` under `--dry-run`), one row
  per arm: `arm  state  rc  seconds  dirty  log  note`, last row wins. Logs are
  `$SESSION/logs/<arm>.log`.
- **Resume** by running the same command with `SESSION=<that directory>`, or
  `--resume-latest`. Rows in `DONE`, `CLAIM_FAIL` or `INVALID` are LATCHED and
  skipped; `REFUSED` and `RETRY` rows are attempted again (a refusal cost
  nothing, and the usual reason to re-run is that its precondition was fixed);
  an `UNKNOWN` row (exit 1 from a file that has not adopted
  `moe/bench/exit_codes`) is not latched and is disclosed by name. Delete a row
  from the ledger to force a latched arm to run again. ONE ARM IS EXEMPT:
  `counter_plan` re-probes on every pass whatever its last row read, because it
  measures nothing and reports a property of the POD. `SESSION_PREFIX` is keyed
  to the card MODEL and `/workspace` is shared between rentals, so a resume can
  be on a different pod, and a latched verdict from the previous one would gate
  the two 120-minute counter arms on silicon nobody asked. Re-asking is about
  fifteen seconds against 240 minutes; the stale row stays as history and the
  gate reads the newest.
- **A RESUME RE-RUNS NO INVALID ROW.** That is the latch, and it means a
  session whose arms landed INVALID resumes into nothing however many defects
  have been fixed since. The 2026-09-09 session landed six INVALID rows and the
  2026-09-10 one landed four. So the driver's own rerun is `--new`, which
  opens a fresh ledger deliberately, and it re-runs an INVALID arm only where
  the defect behind it has actually been fixed. THIS IS THE STANDALONE PATH,
  not the next booking: the next session is the alpha(G) chain (its section
  below). The driver prints the exact command and the state each arm is
  expected to reach under the heading THE STANDALONE DRIVER'S RERUN, AND WHY
  IT IS `--new`, and says first that the chain is the next session; the chain
  runs the driver for its preconditions, so the same block lands in the
  chain's `preconditions.log` as the driver's record, not as the chain's plan:

  ```bash
  bash scripts/h200_gaps_session.sh --new \
    --only thermal,calibrate,pin_probe-n64-g1,bn_g16,dtype,counter_plan,counter-n32-m64,counter-n128-m64,counter_contrast
  ```

  ~303 priced / ~376 bounded minutes. THIS BLOCK CARRIED THE 2026-09-09 SET
  UNTIL 2026-09-10 (calibrate, pin_probe-n64-g1, roofline-n64-g1, cap_test,
  bn_g16, dtype, bm128_depth, alias_ablation at ~71 priced minutes) and all
  eight of those have since been spent, so it was re-booking arms that already
  hold a result. `bn_g16` is expected to reach CLAIM_FAIL again and that is
  its result; each counter arm is expected DONE or CLAIM_FAIL and either is
  the session's headline. THE COUNTER IS TWO ARMS AND THE SET BOOKS BOTH,
  which is what took this block from ~180 priced minutes to ~300: the contrast
  that decides whether the term the session found is TRAFFIC or TIME is
  between BLOCK_N=32 and BLOCK_N=128 at a fixed BLOCK_M, and one cell is not a
  contrast. Booking one of the two spends 120 minutes and answers only the
  half either arm answers alone, which is `alpha_b` as a traffic slope.
  THREE ARMS ARE DELIBERATELY OUT of the set:
  `roofline-n64-g1`, `alias_ablation` and `noise_floor` each hold an INVALID
  whose cause is a gate or an instrument rather than a flag, so re-running them
  buys the same word for the same minutes; the driver names them where it
  prints the booking. Editing rows out of a ledger by hand is the other way
  back and it rewrites the record of what was spent.
- The session's own start is read from the UTC stamp at the END of the
  session directory's name, so a resume into the default directory still
  counts the calibration its first pass published. An operator-supplied
  `SESSION` with no stamp falls back to "now", and a resume under such a name
  refuses until `calibrate` runs again: three minutes rather than a session.
- Every arm is given `MOE_RESULTS_DIR=<root>/gaps-<card>` with the card in the
  path. An operator-supplied `MOE_RESULTS_DIR` that does not contain the card
  is REFUSED, because `/workspace/results` outlives the pod and a second card
  would resume the first card's directories and publish its timings under the
  wrong heading (`moe/bench/provenance.py`, collision 2).

## The calibrate --publish gate

Arm 0 is `scripts/calibrate_hardware.py --publish`. Every arm below it
resolves its ridge, its LEVEL reference clock and its bandwidth through
`roofline.load_measured()`, which reads the TRACKED
`moe/bench/hardware/measured_<card>.yaml`; without `--publish` a calibration
lands only on an untracked path under `results/calibration/` and every later
arm quotes the last rental's ruler while labelling it "measured on this
machine". So after arm 0 the driver REFUSES the rest of the session unless
BOTH halves answer:

| half | question | words |
|---|---|---|
| the yaml | does the tracked file for this card carry a `provenance.utc` at or after this session's own start | `PUBLISHED`, else `MISSING` (no file), `UNDATED` (no provenance block: the committed H200 yaml before 2026-09-02 was this), `STALE` (a previous rental's), `NO_BASELINE` (the driver could not stamp its own start) |
| the arm | does this session's ledger say `calibrate` is `DONE` | anything else refuses, by name: `INVALID` (it measured, failed `clock_established`, and published the yaml BEFORE scoring, so a fresh-stamped file proves nothing), `CLAIM_FAIL`, `REFUSED`, `UNKNOWN`, `NO_ROW` (`--only` left arm 0 out) |

The gate is deliberately NOT scoped to `--only`: nothing that measures is
either. The yaml is the one tracked file the session is meant to change;
commit it unless the `ruler` arm's P1 FAILED (the compute peak was then
sampled in the wrong clock state).

## The arm ledger: what each arm buys

The order is the argument: anything whose result changes how a later arm is
READ runs before that arm. Minutes and clocks are the driver's own cost table
(`--dry-run`), each figure read off the arm's plan; WALL charges compiles and
allocation, KERNEL excludes them in the plan's own words, ALLOW is the
driver's standing allowance where a plan prints no estimate, FREE is an arm
booked at zero because it refuses or times nothing. The last column is the
independent verdict of the 2026-09-03 pod-readiness check (FINAL_VERDICT
section 3) and what the driver did with it. Arm names below are asserted
against `--list` by `tests/test_docs.py`.

| arm | min | clock | what it buys | verdict and disposition |
|---|---:|---|---|---|
| `thermal` | 3 | WALL | CAN THIS CARD HOLD A CLOCK, asked before the ruler is measured on that clock. Sustained dense bf16 GEMM for 30 s of discarded ramp plus a 120 s scored window; C1 is the median SM clock against `timing.THERMAL_FLOOR_FRACTION` of the card's OWN maximum, read from the device, and C2 is the DRIFT rule over the first and last thirds | KEEP, and it REFUSES the session. On 2026-09-11 a rented H200 fell from 1980 MHz to its 345 MHz floor under load, `calibrate_hardware.py` published a tracked ruler reading ridge 73.6 against a real ~156, and its `not_throttled` gate PASSED because it scored DRIFT and a floored card does not drift |
| `calibrate` | 3 | ALLOW | this pod's own ridge and both dtype peaks, published; five arms refuse without it | KEEP: the only file worth committing |
| `pin_probe-n64-g1` | 2 | ALLOW | does `MOE_FORCE_TILE` reach the kernel at BLOCK_N=64, GROUP_SIZE_M=1, the pinning every alpha arm uses | KEEP: precondition for every tile claim |
| `pin_probe-n256-g16` | 2 | ALLOW | the same at vLLM's shipped BLOCK_N=256, GROUP_SIZE_M=16 | CUT in the verdict (it served two refusing rooflines); still booked, 2 min |
| `private-mixtral-bm32` | 10 | KERNEL | ALPHA AS A RATIO OF TWO MEASURED SLOPES, slope(shared)/slope(private), where the private arm gives every M-tile its own copy of the expert weights: no assumed bandwidth and no fitted intercept in it. Run at `--duty 0.25` on both branches, so the ladder's 145 s of kernel time is about 581 s of pod | not in the verdict: ADDED 2026-09-14, rebuilt 2026-09-17 (see "The private-weight reference alone" below). BOOKED 10 SINCE 2026-09-22, off the plan's wall line at `--duty 0.25` plus the probe's 9 s at full duty; it was 3 at full duty, where session 4's four ratio pages were all INVALID on V7, the two arms' clocks 2-18% apart. The row stays KERNEL because the figure still leaves out the compiles and the 25.4 GB weight build |
| `elasticity-m32-n64-g16` | 40 | WALL | the clock elasticity of the per-M-tile cost at one pinned cell (BLOCK_M=32, BLOCK_N=64, G=16), four duty states from the cap to the boost ceiling (1.0 0.5 0.25 0.1); the band it lands in decides what every alpha below MEANS | not in the verdict: ADDED 2026-09-16. THE STANDALONE ARM, session 4's design read on this session's card; the alpha(G) chain measures its own elasticity once per G of the ratio ladder, and that is the chain's step, not this row. `--session-tag` IS ON BOTH BRANCHES SINCE 2026-09-22: without it the run id on an H200 was session 4's own, and on the shared volume the arm would have re-scored session 4's 416 cells as this session's. Its DEVICE guard, also 2026-09-22, now refuses that directory instead, so the tag is what lets the arm measure |
| `roofline-n64-g1` | 1 | KERNEL | THE CONTROL: BLOCK_M=128 at the swept configuration; can refute the ceiling, cannot confirm it for production; its predicted outcome is already NOT TILE-ATTRIBUTABLE | KEEP as the control for `bm128_depth` |
| `roofline-n256-g16` | 0 | FREE | THE CLAIM: BLOCK_M=128 at the configuration vLLM ships. No arm confirms the headline on sm_90: at BLOCK_N=256 no BLOCK_M=256 control fits at ANY warp or stage count (65536 of 65536 registers per block; `bm128_roofline.py --dry-run --block-n 256 --group-m 16 --control 256 --capability 9.0` exits 2), and no BLOCK_SIZE_N does either | CUT: REFUSES at `--capability 9.0`; booked zero, the refusal is the finding. It is not one fix away: `--num-warps 16 --num-stages 3` refuses too, so the paper's headline has no confirming arm on this card |
| `roofline-n256-g32` | 0 | FREE | the same at the swizzle vLLM ships at 2048 tokens | CUT: refuses for the same missing control |
| `bm128_depth` | 5 | KERNEL | five clean memory-bound treads at the production tile, `--r-max 2048`; the whole 128 row currently rests on two fits | KEEP, and the BM=32 scaling partner is not optional: at the pairing {128, 256} the non-vacuity floor is 0.838 of the roof, no BLOCK_M=256 ladder in the corpus reaches it, and the arm refused its own reference and landed INVALID after 292 s on 2026-09-09. THE PARTNER IS NOT A FLAG and this row named one (`--partner-block-m 32`) until now: it is unconditional in `scripts/bm128_depth.py` (`SMALL_TILE_BLOCK_M`, in `BLOCK_SIZES`), so it is in the plan and in the run id without either branch asking for it, and the 5 min books whatever that plan prices (252 s without the partner, 294 s with it) |
| `alias_ablation` | 14 | WALL | THE FIRST INFERENTIAL LINK: is the per-tile slope DRAM traffic at all, measured with no byte model, bandwidth, ridge or intercept | ADDED on the verdict's finding that nothing scheduled it; read its P1 RESULT line's WORD, not its exit code. BOOKED `--dot-fallback refuse` SINCE 2026-09-09: under `allow` the 2026-09-09 run fell to dot mode, spent 308 s and latched INVALID with P1 UNKNOWN at alpha >= 0.229 (ARMS.tsv, rc 3: level, placebo, form and bracket all FAILed after measuring). A probe miss now costs 2.0 min and exits 3; the sum grid it probes was widened downward in warps |
| `noise_floor` | 120 | WALL | a real between-replicate sd, `--replicates 3`, four arms, `--publish` into `results/published/NOISE_FLOOR.json`; until it exists every MDE line says ASSUMED | KEEP only bounded and published, which it now is; the verdict's other precondition (children read on CLAIM_FAIL) is that script's own fix |
| `bn_g16` | 46 | KERNEL | `alpha_a` as a fitted slope and the residual that says whether the three-term model is complete. THE "IT LIVES ONLY IF `alpha_a < 0.17`" READING OF THE BLOCK_M=128 ROW IS GONE SINCE 2026-09-10, and this cell carried it until then: the activation re-read term that `alpha_a` is the coefficient of was refuted on the ladder slope alone (its BLOCK_N dependence must double with BLOCK_M and measures 1.115 +/- 0.003 against 2.000), so the BLOCK_M=128 row now turns on `alpha_b` instead, at a threshold of 0.784, and the counter arm is what measures it | KEEP, and booked 46 since 2026-09-10, not 36. The swept set gained a THIRD subject tile: `--tiles 16,32,64,128`. At {32, 64, 128} on 2026-09-10 the BLOCK_M=128 branch was discarded (its memory branch came within 15% of its compute branch) and six cells over TWO heights were left to fit two parameters, at which every candidate extra term correlates +0.72 to +0.98 with the activation column. BLOCK_M=16 is the added height because its ladder is memory-bound end to end on this card. The plan goes from 1428 timings and 2142 s to 1836 and 2754 |
| `anchor_measure` | 5 | WALL | the memory-branch level measured at matched reuse rather than extrapolated | KEEP |
| `anchor_rescore` | 0 | FREE | every committed report re-scored under the fresh anchor, written under the session directory, never into `results/published` | KEEP |
| `occupancy` | 23 | KERNEL | does alpha track residency or program order; P2 is EXPECTED to fail and that FAIL is the finding | not in the verdict's KEEP table; booked with `--fail-on-gate` so the failing claim reaches the ledger as one |
| `mma_switch` | 7 | ALLOW | whether the tile alone selects the instruction at fixed tokens | KEEP |
| `ruler` | 2 | WALL | prices the read-vs-triad and clocks-first ruler changes on the committed corpus without adopting them | KEEP |
| `cap_test` | 5 | KERNEL | BLOCK_M=16's cap, DEMOTED: on uniform routing vLLM runs 16 multi-tile in 1 of 24 cells, so this tests the formula | KEEP, booked `--r-max 2112`: at the default r_max comes from `depth.rows` (688 on the H200 band), the grid stops at 672 with two exactly-full BLOCK_M=256 stacks against V1's three, and the arm is unsatisfiable from its own plan page. This row read `--r-max 1024` until now, which the script's own V4 check REFUSES at plan time (it wants a 132-tile BLOCK_M=16 stack, 1024 gives 66, and the refusal prints `raise --r-max to at least 2112` and no cost line). 2112 is that printed minimum: 162 cells and 242 s |
| `dtype` | 8 | KERNEL | how much of the 1.15 fp8/bf16 crossing is the config vLLM resolved per dtype | CUT in the verdict until its C3 window is re-derived (at the corrected spread the window has no discriminating power); still booked. RE-SCOPED 2026-09-09: the cross-config arm transplants BLOCK_SIZE_M and GROUP_SIZE_M only, because the full fp8-config-at-bf16-width transplant is infeasible on sm_90 at 22 of 28 cells and, when it was run, vLLM 0.27.1's `override_config` (no try/finally) leaked the fp8 config process-wide and corrupted 41 arms |
| `span_dense` | 31 | KERNEL | the 0.563 extent-versus-kernel split on the dense grid, run WHOLE: `--max-minutes` was removed because it scored a truncated grid as complete | CUT in the verdict until truncation is a refusal; the driver books the honest time instead |
| `span` | 0 | FREE | the same on the published grid, `--no-densify`; refuses on `c2_grid_power` before spending a minute | CUT: booked zero, the refusal is the answer |
| `counter_plan` | 1 | WALL | whether a DRAM counter route is open on this box; BLOCKED is the ANSWER on a rented pod, not a broken instrument | KEEP, and it now GATES the two arms below. TWO THINGS IN THIS ROW WERE STALE UNTIL 2026-09-10. It said the script files BLOCKED as INVALID: since 2026-09-03 `dram_counter_route.py --probe` scores one gate per verdict and exits through `exit_codes.classify`, so OPEN is DONE and BLOCKED is CLAIM_FAIL. And it said do not act on the printed ncu recipe, which was the right instruction while every rented pod refused the counter; on 2026-09-10 the probe read OPEN, so acting on it is what the two `counter-*` arms are. **THAT OPEN IS RETRACTED (2026-09-15)**: the probe profiled `/bin/true`, which launches no CUDA kernel, so it never attempted a counter read and could not fail, and a rented H200 then refused the read with `ERR_NVGPUCTRPERM` after two 120-minute arms had been booked on the word. The probe now launches a real kernel and reports OPEN only when a counter value comes back; expect BLOCKED, and read the `counter-*` arms as conditional on the probe rather than as planned work |
| `counter-n32-m64` | 120 | WALL | THE COUNTER RUN, added 2026-09-10, and it is TWO arms: DRAM traffic at BLOCK_N=32 and at BLOCK_N=128, both at `--block-m 64`. It is the only instrument in the study that reads BYTES, and `alpha_b = (dR/dn - a_per_tile)/W` from either arm has no fitted level, no `delta`, no `D` and no assumed bandwidth: today the same six `bn_g16` cells give 0.6087, 0.5930 or 0.5143 depending only on which rate is assumed. The BLOCK_N CONTRAST, which needs both arms, is what decides whether the term that replaces the refuted activation re-read is TRAFFIC or TIME: 3.85 GB against 2.06 GB of weight-set-equivalent per M-tile at BLOCK_M=64, or the same bytes at both | RUN THEM ONLY WHEN `counter_plan` READ OPEN; a BLOCKED probe retires both for the whole session in about fifteen seconds. 120 WALL min is the plan page's own "two pod-hours end to end" for ONE set of 12 profiled invocations, and the COST block is byte-identical at both BLOCK_N, so the pair is 240. THIS ROW WAS ONE ARM BOOKED 120 UNTIL NOW and it described the contrast anyway: the arm line passed no `--block-n` and no `--block-m`, so it ran the script's defaults of 64 and 32, the single pinned cell the 2026-09-10 analysis named as a defect to fix before running. GROUP_SIZE_M is still pinned at 16 on both arms; a G=1 cell is a third arm and is not booked. The driver checks that `dram_counter_route.py` defines `--run` and `skip_arm`s both rows by name if it does not, rather than spending an argparse exit 2 on the card |
| `counter-n128-m64` | 120 | WALL | the same at BLOCK_N=128; it is the other half of the contrast above and neither half is readable without it | the same gate, the same booking, one `counter_arm` call site for both so a fix cannot land on one of the two |
| `counter_contrast` | 0 | FREE | THE READING THE PAIR IS PAID FOR, added 2026-09-10 after a build audit found that `grep -rn -- --contrast` over the driver and these docs returned nothing: the session booked four pod-hours to write two payloads and then left the ratio between them to the operator, by hand, off the printed predictions. `--contrast` scores it: TRAFFIC predicts 1.871, TIME predicts 1.000, 87% apart, each scored at +/-5%, and either word is a result | ADD IT. Zero GPU minutes, over the two files the pair writes. NOT_PLANNED off a GPU box, because `--contrast` is exclusive with `--dry-run` and neither payload exists yet; SKIPPED rather than scored when only one payload is on disk, since a claim gate over one cell would file a missing run as a refutation |

`bn_g1` was DROPPED, not demoted: at GROUP_SIZE_M=1 that script's own design
self-test exits INVALID (S4's `saw sd` of `alpha_a` 0.1562 against a gate of
0.025) and
the same self-test fails at every pinning checked except 16, so there is
nowhere to re-pin it to.

**What a rental reaches.** The driver prints, from its own table, that
~583 priced minutes (~144 of them KERNEL, which exclude compiles) become
~778 once the KERNEL part is multiplied by the one wall-over-model ratio this
repository has measured (2.35x, mixtral_g1 on the s4 arm); that second number
is an illustration of the gap, not an estimate of any arm. THOSE THREE
FIGURES READ 262 / 107 / 407 UNTIL 2026-09-10 and none of the three had been
true for two sessions: the whole-session total moved when `bn_g16` went from
36 to 46 priced minutes and again when the counter pair added 240, and this
paragraph was retyped from a `--dry-run` that predates both. THEY THEN READ
515 / 119 / 676 UNTIL 2026-09-22, which predated four arms (`thermal`,
`private-mixtral-bm32`, `elasticity-m32-n64-g16`, `blockk-w4`) and the
private reference's move to `--duty 0.25`, which took it from 3 booked minutes
to 10. Re-read it off `bash scripts/h200_gaps_session.sh --dry-run` rather
than from here; the driver computes it from `arm_minutes` and `arm_clock` and
this page is a copy, which `tests/test_h200_gaps_session.py` now checks.

A two-hour rental reaches both payload arms, their preconditions and the
private-weight reference
(`--only thermal,calibrate,pin_probe-n64-g1,pin_probe-n256-g16,private-mixtral-bm32,roofline-n64-g1,roofline-n256-g16,roofline-n256-g32,alias_ablation,bn_g16`);
a three-hour one adds `bm128_depth`, the anchor pair and the cheap tail.
NEITHER contains the noise floor: it is 120 WALL minutes on its own and a
rental that enters it without finishing it fails that script's own V2 and buys
nothing. Book it on its own.

**Read these first** when it ends: the `private-mixtral-bm32` ratio and the
`elasticity-m32-n64-g16` band (neither measures an alpha below; both decide
what every one of them means), then the `alias_ablation` P1 line, the
`roofline-n256-g16` verdict line (or its refusal), the `noise_floor` sd, and
the `bn_g16` residual line. The driver prints them under that heading.
`elasticity-m32-n64-g16` is the driver's STANDALONE elasticity: session 4's
design (G=16, duty 1.0 0.5 0.25 0.1) under this session's `--session-tag`, a
second card's reading of it cell for cell. It is not the alpha(G) chain's
elasticity, which is measured once per G of the ratio ladder and is the
chain's own step.

**What to commit.** `moe/bench/hardware/measured_<card>.yaml` and
`results/published/NOISE_FLOOR.json`, both written under `--publish`, both
tracked; nothing else lands in the tree. To publish an arm, copy its run
directory under `results/published/<date>-<gpu>-<arm>/` and run
`git check-ignore -v <path>`, which must print nothing. The alias ablation has
no `--publish` because it has no tracked file to land in; copy its directory
whichever way its P1 line reads, since PASS and FAIL both decide the paper's
mechanism sentence.

---

# Part B: the companion to `scripts/pod_session.sh` (the 2026-09-01 sweep session)

That script is the sweep session; this part says what each of its gates MEANS,
so a tired person at 02:00 can act on a FAIL without rereading
`docs/FINDINGS.md`. It predates the apparatus rebuild: its step scripts speak
`[PASS]` / `[FAIL]` lines and its own exit codes (0 / 1 / 2 / 3 at the end of
this part), not the `RESULT:` line and the one table of `docs/APPARATUS.md`,
and several of its registered predictions have since been retracted; each is
marked inline below.

**The whole session is one command.**

```bash
cd /workspace/repo
bash scripts/pod_session.sh --label alpha-0558
```

It prints PASS or FAIL against a NUMBER at every step, says on each FAIL whether
to continue or stop, and refuses to call the session finished until it has
verified that everything worth keeping exists somewhere that outlives the pod.

---

## The private-weight reference alone (2026-09-17)

The one arm whose number needs no assumed bandwidth and no fitted intercept
(`private-mixtral-bm32`, 10 min) was rebuilt on 2026-09-17 after two reviews.
THIS SECTION IS THE STANDALONE PATH, R3 (this arm,
`scripts/private_weight_reference.py`) alone through the driver on a short
rental; the next session is the alpha(G) chain below, which runs R3 at every G
with R1 (`scripts/clock_elasticity.py`) beside it. The section was written to
have R3 run BEFORE the other two arms it shares `three-arms` with
(`elasticity-m32-n64-g16`, `blockk-w4`), because the 2026-09-17 reviews found
design defects in both. R1's, in the script behind `elasticity-m32-n64-g16`,
were fixed on 2026-09-22: its gated claim is the per-M-tile elasticity over
treads 2 and deeper, with tread 1 printed beside it off the law; a resume on
another card of the same name is refused on the card's UUID; and its plan page
prints the claim's own resolution. blockk-w4's review is not recorded in this
repo and was not redone; since it, the script's kernel probe (2e2f1d8,
2026-09-21) and its `--card` check (6f3a6ce, 2026-09-22) were fixed, and
`git log -- scripts/blockk_diagonal.py` is the full list of its changes, not
this sentence. Rent about an hour and run only the preconditions and this
arm; the driver prices the four at ~18 minutes (~32 bounded), and the
hand-run second seed below is about 10 more:

```
bash scripts/h200_gaps_session.sh --new --only thermal,calibrate,pin_probe-n64-g1,private-mixtral-bm32
```

What the rebuilt arm does that the plan page states before a byte is timed:
copy `c` of expert `e` sits at expert slot `e x 9 + c` (expert-first, so the
private arm's tiles run in the shared arm's order), SHARED and PRIVATE both
declare all 72 slots at every tread (one sorted-id buffer, one launch grid,
one dead-launch count, so the two differ ONLY in the addresses tiles read),
and NATIVE is the study's own 8-expert call over a strided view of copy 0.
Nine copies are declared and six read: 25.4 GB of a 141 GB card, and the
padding is what keeps both ratio arms on ONE `moe_align_block_size` kernel,
because vLLM switches kernel at ids < 1024 and experts <= 64 and this ladder
crosses the id bound between treads 3 and 4. Before the weights are built the
arm times the alignment op alone along the ladder, once per arm (NATIVE's
declaration, and SHARED's and PRIVATE's id sets at the ratio arms'); V8
refuses the design on that measurement and SKIPS the sweep when it comes back
FAIL (a step over budget, resolved against its own standard error), and ONLY
then. An UNKNOWN (over budget but unresolved, or an eager host-bound probe that
did not resolve NATIVE's own switch at the census tread) is a statement about
the instrument, not the design: the ladder still runs, and the page latches
INVALID on V8, a VALIDITY gate, with every other gate's number beside it. The
probe times the op under a CUDA graph (`PROBE_CALLS_PER_REPLAY` calls per
replay), so its cells are GPU time on the H200 too: session 4's eager probe
was host-bound 36 of 36 (32-36 us of host per call against a kernel of a few
us) and read UNKNOWN. NATIVE's switch is read beside the ratio series:
resolved at the census tread it confirms the cited source on this build;
unresolved it bounds the switch under the printed threshold per call. Only the
eager fallback (capture refused, named on the page) needs the control to earn
PASS. NATIVE keeps the switch, and V5 fits it out.

Read, in this order: V8 (one kernel along the ratio arms' ladder, measured),
V7 (the two arms' clocks agree at every tread; below), V2 (each copy read by
exactly its own tiles, one copy zeroed at a time), V5 (the declaration's
per-tile cost with native's step out; the step itself is printed with an
interval), V6 (shared and private agree at n=1, where they are the same
call), then C1, which names the world the ratio landed in: ISSUE-AND-LATENCY,
below/at/above the refit band, or NO-REUSE. C1 UNKNOWN means the point and the
interval disagree on a world and the claim is unresolved at this precision,
not that the arm broke. The arm REFUSES at plan time if the under-load clock
sampler cannot read the card (V7 would be UNKNOWN throughout), and it writes a
`DEVICE` file with the GPU UUID under its results directory so that a resume
on another pod of the same card type is refused rather than merged.

**V7 at the duty the driver runs.** V7 is expected to hold at 0.25, the duty
the driver runs this arm at (why, below). A V7 FAIL at 0.25 means that duty is
not yet flat for that arm on this card; the page names a lower duty; the chain
skips the G's later seeds and prints the follow-up command. On this section's
standalone path the follow-up is by hand, as the pair below says: a new design
key and its own runs at the lower duty, read with `--read` on the laptop. At
full duty, on a pod that cannot lock its clock, V7 FAILS by construction
whenever the arms draw different power and the page is INVALID, which is why
the arm is duty-cycled, below.

**The clock-corrected ratio.** `--clock-elasticity ETA LO HI` takes R1's
per-M-tile claim: `elasticity.value`, `elasticity.lo` and `elasticity.hi` of a
clock_elasticity report whose `elasticity` block carries `claim_min_tread`,
the claim fitted over treads 2 and deeper. Session 4's committed reports carry
no `claim_min_tread`, and their `elasticity.value` is the pooled per-call
reading, which is not that quantity: do not pass it. Name the source the way
R3's `--help` asks, with the report, the key read and the commit:
`--clock-elasticity-source '<report>:elasticity.value@<sha>'`. C1 then PRINTS
a clock-corrected ratio beside the raw one, scored by nothing.

One thing to eyeball on the pod before the run, because V2 would only catch
it after the sweep: the installed vLLM's `fused_moe.py` must cast
`off_experts` to int64 (slot 71 x 117 MB is past 2^31 bytes from the weight
base).

**The driver runs it at `--duty 0.25`**, on both branches, and the alpha(G)
chain runs R3 at the same duty: DESIGN DECISION 15 as the owner decided it on
2026-09-22. The script's own default stays `--duty 1.0`: duty is a design key
defaulting to 1.0, and moving the default would break `--replicate-of` against
session 4's runs. WHY A DUTY AT ALL: at full duty the power cap boosts
whichever arm reads fewer bytes, so on a card that cannot lock its clock V7
fails by construction, and session 4's four ratio pages were all INVALID on V7
that way, the private arm's clock 2-18% below the shared arm's (2% on
mixtral-8x7b at G=1 in both of its runs, 12% on qwen2-57b-a14b at G=1, 18% on
mixtral-8x7b at G=16). `--duty D` times each cell as bursts of about 40 ms of
kernel time separated by idle gaps, which takes board power off the cap
without changing a byte the kernel moves. WHY 0.25 AND NOT 0.5 is session 4's
clock arm, the same native kernel under the same `time_duty` R3 imports: at
duty 0.5 the clock still tracked board power (-1.09 MHz/W over 1882-1965 MHz,
13 of the 78 cells at treads 1-6 drifting, 16.7%); at duty 0.25 every tread's
median sat at 1965 MHz, at 277-317 W, and no cell drifted. So V7 is expected
to hold at 0.25, not guaranteed: the private arm is inferred to draw more
power than the kernel that was measured (on session 4's cap it held 1425 MHz
where the shared arm held up to 1740 at G=16), and its V7 line, which now
prints each arm's power, is the measurement. The price is
wall clock, about 1/0.25 = 4 times the kernel time: the plan prices the
ladder's 145 s of kernel time at about 581 s, so a run is about 10 minutes
before its compiles and the 25.4 GB weight build. The duty is in the run id,
and board power is recorded per cell.

**Book this arm as a PAIR.** The interval on C1 is a bootstrap over repeats
within one run; on 2026-09-21 two G=1 runs at seeds 0 and 1, 77 minutes apart
on one pod, read ratios whose intervals did not overlap. The driver runs seed
0; the second run is by hand after it: the driver's own measuring line
(section 0c of `scripts/h200_gaps_session.sh`) with
`--seed 1 --replicate-of <seed-0 run dir>/report.json` added. The seed-0 run
directory is the `WRITES TO` line of `$SESSION/logs/private-mixtral-bm32.log`,
and the session tag is the session directory's name:

```
MOE_RESULTS_DIR=/workspace/results/gaps-<card> /workspace/venvs/vllm/bin/python \
  scripts/private_weight_reference.py --model mixtral-8x7b --block-m 32 \
  --treads 6 --repeats 9 --duty 0.25 --session-tag <session directory name> \
  --seed 1 --replicate-of <seed-0 run dir>/report.json
```

Keep `--duty 0.25` and `--session-tag`: duty is a design key, so
`--replicate-of` refuses a replicate at another duty. C1 is then scored on
the envelope of both runs' intervals and the page prints the cross-run
spread; a lone run's page says it was scored alone.

**After a V7 FAIL at seed 0**, a seed 1 at 0.25 re-measures the same clock
split, which is why the chain skips a G's later seeds on it; skip the
hand-run seed 1 too. The follow-up is the line above at `--duty 0.1` (the
page names a lower duty and no number; 0.1 is the one the chain's follow-up
command registers): first as its seed 0, ending the command at
`--session-tag <session directory name>` with no trailing backslash, then
with `--seed 1 --replicate-of` that run's report. It is a new design
key and its own pair, outside the driver's ledger, and it is read with
`--read` on the laptop like any pair.

A pair already on disk is re-read on the laptop with

```
.venv/bin/python scripts/private_weight_reference.py --read RUN1/report.json --replicate-of RUN0/report.json
```

and that output, not a hand-computed difference, is the figure to quote.

---

## The alpha(G) chain (2026-09-22): the next session's command

The next pod session is one sequenced ledger, not the arm by hand.

**Bring the pod's checkout to the pushed head first.** The volume's clone
carries the last session's ruler yaml modified (calibrate `--publish` writes
the tracked file), and git refuses to switch branches over it even when it is
byte-identical to the committed copy:

```
git -C /workspace/moe-kernels checkout -- moe/bench/hardware/measured_nvidia_h200.yaml
git -C /workspace/moe-kernels fetch origin
git -C /workspace/moe-kernels checkout -B r3-align origin/r3-align
git -C /workspace/moe-kernels log -1 --format=%h    # must print the head that was pushed
```

Compare that hash with `git log -1 --format=%h origin/r3-align` on the laptop;
do not type one from memory.

**Launch it detached, appending.** A dropped ssh session kills a foreground
chain and the arm in flight. The console APPENDS (`>>`): a `--resume`
launched the same way keeps the earlier passes' lines (the lock's fallback
notice, the STOP explanations, the per-run lines, which no ledger holds), and
the exfil line carries the file. Watch the log and the ledger:

```
cd /workspace/moe-kernels
bash scripts/alpha_g_chain.sh --dry-run       # plan and price; nothing measured
nohup setsid bash scripts/alpha_g_chain.sh >> /workspace/alpha_g_chain.out 2>&1 < /dev/null &
tail -f /workspace/alpha_g_chain.out          # and $SESSION/CHAIN.tsv, one row per step
```

Continuing is `--resume` (the newest `alpha_g-<card>-*` directory that holds
`CHAIN.tsv`, never a dry run's) or `SESSION=<dir>`, launched the same
detached, appending way. A bare run refuses when a chain session for the card
already holds `CHAIN.tsv`, and prints both commands; `--new` opens a fresh one
on purpose. `--dry-run` with `--resume` or `SESSION=` is refused: it would
overwrite that session's step logs, and the driver's `logs/thermal.log`, with
plan pages. A measuring run is refused on a box whose card torch cannot name,
with the probe's reason and nvidia-smi's name, driver version and power limit
printed. It holds a lock on the session for the whole run, and a second chain
on it is refused. On the pod the lock is a `flock`, and a held one is never
taken over, whatever pid it records: when the recorded chain is dead, the
holder is the arm it left running (`pkill -f alpha_g_chain.sh` kills the
shell, not its python), still timing the card. The refusal prints `fuser -v`
and `lsof` for the lock file, and a line that needs neither, since the image
may carry neither:

```
ps -eo pid,etime,args | grep -E '[a]lpha_g_chain|[p]rivate_weight_reference|[c]lock_elasticity|[h]200_gaps_session'
```

Stop what it names, then `--resume`. Only when that line names nothing on this
pod is the holder on another pod sharing the volume; do not open a `--new`
session meanwhile, since its arms would time the card beside whatever holds
the lock. Only the `mkdir` fallback, on a box without `flock`, lets `--resume`
take over a lock whose pid is dead or whose host differs. On its first
measuring pass the chain records the card's UUID in `$SESSION/DEVICE` and R3's
duty in `$SESSION/R3_DUTY`, and refuses a resume on another card or at another
`R3_DUTY` (a session at another duty is `R3_DUTY=<d> ... --new`). The DEVICE
check refuses before any step runs; R3's DEVICE file and R1's `device_guard`
each refuse a run directory measured on another card as well, so either layer
alone refuses a replacement pod's card. After a pod is lost, the replacement
pod is `--new`.

**An override holds.** `--past-gpu-tests` and `--past-v8` write the decision to
the ledger as an OVERRIDDEN row (`gpu-tests-override`, `probe-check-override`,
`v8-override`), and it holds for every later pass of the session: a plain
`--resume` after it goes on past the same gate without the flag, says so on
the console, and runs neither `tests/test_gpu.py` nor the probe check again.

**Every arm step is capped.** The probe check, R1 and R3 each run under
`timeout --signal=INT --kill-after=60`, as the pytest steps do, at max(3 x the
arm's own `--dry-run` price, 30 min), the plan run again just before the step
(the probe check prints no plan: its 120 s allowance puts it at the 30-minute
floor, and so does the counter probe's price, the driver's own booking for
`counter_plan`). A timed-out arm is ERROR with TIMED OUT in its note, not latched: both
arms resume per cell, so `--resume` re-runs it. A process parked inside the
volume's FUSE request cannot be signalled, by this or by anything.

It runs, in order:

1. Both arms' `--self-test`. Each must be DONE: an INVALID self-test is a
   scorer that failed its own planted world. One that is not DONE runs again
   on every pass, in seconds, so after the scorer is fixed and checked out on
   the pod a `--resume` re-proves it.
2. The driver's thermal, calibrate and pin_probe-n64-g1 in the chain's own
   session directory. The chain stops when the driver exits 2 (its thermal,
   calibration or reference-grade gate refused), and unless thermal and
   calibrate are DONE in its `ARMS.tsv`. The step latches only on the
   driver's exit 0, so a row it still owes is re-attempted on `--resume`.
   pin_probe-n64-g1 runs for the record and is not gated: it asks whether
   `MOE_FORCE_TILE` reaches the kernel, and R1 and R3 both pin through vLLM's
   `override_config` and never read `MOE_FORCE_TILE`.
3. The counter probe, INFORMATIONAL: never gated and never latched. It
   decides whether the one route to alpha at G >= 2 at this tile, `ncu`
   bytes (item 7 of session 5's findings), can happen on RunPod at all. It
   finds `ncu` on PATH, then at `NCU_SEARCH`'s globs
   (`/usr/local/cuda*/bin/ncu` and `/opt/nvidia/nsight-compute/*/ncu` by
   default, each one's matches newest name first), and runs
   `scripts/dram_counter_route.py --probe`, the driver's `counter_plan` probe
   and not a second copy of it, from PY_BASE with the first one's directory
   first on PATH: one real kernel under ncu, and `dram__bytes_read.sum` read
   back or refused. Its ledger row's state is `INFO`, which no pass latches,
   so it runs on every measuring pass (a counter route is a property of the
   pod). The note starts with the verdict (OPEN, BLOCKED, ABSENT, UNTESTED or
   ERROR) and carries ncu's path and version, the exact error line and the
   two capabilities (`CAP_PERFMON`, `CAP_SYS_ADMIN`); `COUNTERS` in the
   session directory holds the same, with the probe's own notes, beside its
   payload, `COUNTERS.json`. When ncu is found off PATH, `COUNTERS` says so:
   the driver's `counter_plan` and `dram_counter_route.py --run` look on PATH
   only. A dry run prices it at the driver's own `arm_minutes` for
   `counter_plan` and writes a SKIPPED row. RunPod's record, as this repo
   commits it: two rented H200s attempted a counter read and both were
   refused with `ERR_NVGPUCTRPERM`, on 2026-08-25 (`ncu` over the harness's
   own CLI, `profiles/q2_kernel_names.txt`) and on 2026-09-15, on a pod
   holding neither `CAP_SYS_ADMIN` nor `CAP_PERFMON`; the 2026-09-09 and
   2026-09-10 pods were never asked (their probe profiled `/bin/true`);
   session 4's probe (2026-09-21) read `no ncu on PATH` and looked nowhere
   else; session 5 attempted none. Two refused pods are a record, not a fact
   about the platform: the probe answers for the pod it runs on.
4. `tests/test_gpu.py` on the card, from PY_BASE: the timing and clock
   primitives both arms stand on. It does NOT cover the graph probe R3's V8
   stands on: that one test imports vLLM and skips from PY_BASE, which is why
   step 5 exists. Not green stops the chain, and an exit 0 in which no test
   passed is not green (off a card every one of them skips).
   `--resume --past-gpu-tests` goes on after you have read the failures, and
   the ledger records that decision as its own row.
5. The probe check, `private_weight_reference.py --probe-check --model
   mixtral-8x7b --block-m 32` from the vLLM venv: that skipped test's on-card
   check alone (`moe_align_block_size` captured under a CUDA graph,
   `PROBE_CALLS_PER_REPLAY` calls per replay, not host-bound, the graph's
   per-call time under the eager p50), about a minute and one RESULT line,
   before the pilot spends a ten-minute ladder finding the same thing. Not
   DONE stops the chain and quotes the page's RESULT or REFUSED line, and it
   runs again on every pass until it is DONE. After INVALID, UNKNOWN or ERROR
   the remedy is a code change (`PROBE_CALLS_PER_REPLAY`, or the capture)
   brought to the pod's checkout, then `--resume`; `--resume --past-v8`, the
   same instrument as the pilot's V8, goes on, recorded. REFUSED timed
   nothing: the check found no card, no vLLM, or a vLLM op that did not
   import, so the interpreter or the card is wrong, not the probe. The STOP
   names the `PY_VLLM` in use (it falls back to PY_BASE when the vLLM venv's
   python is missing); check that it imports vllm and that its torch wheel
   matches the driver, then `--resume`. It does not offer `--past-v8` there:
   R1 and R3 run from the same interpreter, and the ledger would hold that
   override for every later pass.
6. `private_weight_reference` at `--duty 0.25`, seed 0, at every G of
   {1, 4, 16, 64}. On session 4's clock arm duty 0.25 sat flat at 1965 MHz
   with no drift, and duty 0.5 still tracked board power (-1.09 MHz/W over
   1882-1965 MHz). The chain stops after `r3-g1-s0` unless its V8 is PASS,
   and a missing report.json stops it too: V8 is the alignment probe under
   the graph, it describes the instrument and not G. The STOP prints the
   pilot's `align_probe` note and `graph_calls`, and says what going on buys,
   priced off the arms' own plans: V8 UNKNOWN or FAIL makes every later ratio
   page INVALID (the probe re-runs on every page); `--resume --past-v8` buys
   R1's four regime words (about 75 min) and eleven ratio pages that cannot
   be quoted (about 120 min); `SEEDS=0 bash scripts/alpha_g_chain.sh --resume
   --past-v8` limits the ratio pages to seed 0 (three, about 33 min). The
   ledger records `--past-v8`, and later passes hold to it.
7. `clock_elasticity` at each G with three duty states (1.0, 0.5, 0.25: the
   card on its power cap at 1.0, off it at 0.5 and 0.25), the per-M-tile
   elasticity gated. These are the owner's states since session 5: its G=1
   run at 1.0, 0.7, 0.5 excluded 22.4% of its rows for in-burst clock drift
   (0.7: 30.8%, 0.5: 36.5%, 1.0: 0%) and failed V4, which holds the excluded
   rows to 20%; at 1.0, 0.5, 0.25 every G passed V4 (G=1 16.3%, G=4 6.7%,
   G=16 5.4% and 5.8% on its re-run, G=64 5.8%). `R1_DUTY=` sets others.
8. Seeds 1 and 2 at every G, each scored with the earlier seeds of its G
   through `--replicate-of`. With R1 between seed 0 and seed 1, a G's seed 0
   and seed 1 start about 118 min apart and its seed 1 and seed 2 about 43
   min apart at the dry run's prices of 2026-09-23; the dry run prints the
   spacing off its own prices. A G whose seed-0 page did not read V7 PASS gets
   a SKIPPED row for seeds 1 and 2 (not latched), which are not run, and the
   note is worded by that verdict:
   - V7 FAIL: the two ratio arms ran at different clocks at duty 0.25 (the
     power state), so a later seed at 0.25 would buy the same split. A V7
     FAIL at 0.25 means that duty is not yet flat for that arm on this card;
     the page names a lower duty; the chain skips the G's later seeds and
     prints the follow-up command. The follow-up is that G's three seeds at
     `--duty 0.1` with the chain's own R3 flags and session tag, seed 0 first
     and seeds 1 and 2 with `--replicate-of` its report, written to
     `$SESSION/followup-g<G>.txt` and printed once on the console, priced off
     R3's own plan at 0.1 (about 77 min a G at 2026-09-22's prices). Run it
     AFTER the chain has finished, never beside it. It is a new design key
     and its own runs, outside the ledger and `PAIRS.tsv`, read with `--read`
     on the laptop. The chain does not act on a V7 FAIL by itself.
   - V7 UNKNOWN, absent or unreadable: seed 0's V7 could not be scored
     (nothing timed, e.g. the sweep was skipped on V8, or a clock was
     unread), so a later seed would read the same; the note names the seed-0
     log.
9. The whole suite, only on `END_SUITE=run`: it is off by default, the
   owner's decision in session 5. On that pod the base-venv suite exercised
   nothing the arms depend on beyond `tests/test_gpu.py`, which step 4 runs
   either way, and it ran about 1.4 s a test off the network volume (2499
   tests in 3472 s before it was interrupted at 49%). Without it
   (`END_SUITE=skip`, the default) the step writes a SKIPPED row saying the
   suite was not requested; any other value is refused before a session is
   opened. With it, the suite runs uncapped, from PY_BASE (`-rfE
   --durations=25`), after every arm: a record of the box that gates nothing.
   A suite that already ran to its tally, green or red (pytest exit 1), is
   not bought again, and `END_SUITE=skip` writes no row over it; a timeout,
   an interrupted run or a log with no tally runs again. Both pytest steps
   run without the chain's own knobs in their environment (`SESSION`,
   `END_SUITE`, `G_LADDER` and the rest of `CHAIN_KNOBS`): the suite's tests
   spawn the chain, and a `SESSION=<dir>` launch would otherwise steer them
   into the real session.

**The regime word per G**, read off R1's interval through the arm's own
`band_of`, and what the ratio beside it reads as, which both tables print as
`reads_as` (the rule session 5's findings support): RAW-STANDS (wholly below
0.25: the ratio beside it is a re-read fraction, the one word that reads
`re-read fraction`); UNREGISTERED-GAP (wholly inside [0.25, 0.40]: neither
registered consequence is licensed, so quote the interval and no word;
`unresolved`); CLOCK-CARRIES (wholly above 0.40: the ratio is not alpha. In
session 5 at G >= 4 the shared arm sat on a per-tile floor that scales with
the SM clock, any alpha in [0, 0.60] fit it equally, and the bytes-rate bound
below still proves real reuse there: `blend (traffic and a clock-scaled
on-chip floor): not alpha`); STRADDLES (the interval crosses an edge: no word;
`unresolved`); `withheld:<EXIT>` (R1's page exited INVALID, REFUSED, ERROR or
unscored: no word is read off a page its own gates did not stand behind;
`unresolved`); `unmeasured` (no R1 report for that G yet; `unresolved`). Session
4's G=16 claim over treads 2 and deeper read a half-width of 0.084 over its
states 1.0, 0.5 and 0.25 (0.092 over all four; the all-tread reading's was
0.076) against R1's 0.075 target, half the gap band's width. Session 5 ran R1
at those three states, the chain's, at every G and read half-widths of 0.0096
at G=1 (STRADDLES at 0.40), 0.0660 at G=4 (CLOCK-CARRIES), 0.0837 then 0.0909
at G=16 (withheld:INVALID both times, on V5 and then on V7) and 0.0875 at
G=64 (CLOCK-CARRIES): its pages 0d8858eb, e1c429b7, a5a8fde2.first-v5-invalid
then a5a8fde2, and a3d5cd3a under
`results/published/2026-09-23-nvidia_h200-session5/results/gaps-nvidia_h200/clock_elasticity/`.
An interval within its half-width of 0.25 or 0.40 straddles. The word is a
secant between the capped clock at duty 1.0 and the clocks at 0.5 and 0.25,
and R3 runs at 0.25, at the ceiling, the top of that secant's range. To first
order, for a per-tile cost A + B/f with A and B not negative, the local
elasticity B/(Af + B) lies in [0, 1] and falls as f rises, so RAW-STANDS
carries over to R3's operating point and CLOCK-CARRIES is only an upper bound
there. That form cannot produce an elasticity above 1, which session 4's G=16
claim read at every subset of its states: where R1's point is above 1 the
A + B/f reading does not apply, and the interval is quoted without it.

**What it leaves.** `$SESSION/CHAIN.tsv` is the ledger. The tables are rebuilt
from the reports on disk at the end of every pass and before every STOP, so a
resume never duplicates a row and an R1 that lands on a later pass is joined;
`$SESSION/PAIRS-README.txt` is their legend. `$SESSION/PAIRS.tsv` has one row
per ratio run: G, seed, ratio, within-run interval (R3's 90% bootstrap over
repeats), exit word, `exit_scope` (`alone` when C1 was scored on the run's own
interval, `envelope` when on the envelope with the earlier seeds, so seed 0
and seed 1 of one G can differ in exit by scope, not by result), duty, run
id; the joint reading on that run's page (`rep_n`, spread, sd, envelope,
`joint`), filled only where the run formed a ratio and is in the reading;
each arm's median clock over the ladder and the count of LEVEL LOW (arm,
tread) cells; and R1's eta, 95% interval, word and exit. `joint` is C1's
verdict against the refit band, not a quotability flag: NO-REUSE, expected at
G=1, reads FAIL. `$SESSION/PAIRS-by-G.tsv` has one row per G: every seed of it
that formed a ratio, read together by R3's own cross-run machinery whatever
order the seeds ran in (n, seeds, mean, sd, envelope, joint verdict, any seed
inside the envelope whose own page exited INVALID), beside R1's columns. Quote
PAIRS-by-G.tsv: PAIRS.tsv's joint columns are what each page said when it
ran, over the seeds before it. Both tables carry `reads_as`, what the ratio
can be read as off R1's word (above). PAIRS-by-G.tsv also carries the
bytes-rate bound, the one bound on alpha that needs no private arm (session
5's findings, 3.6): alpha <= (t x C / W - 1) / (n - 1), with t the shared
arm's time at its top tread n off the reports' own ladders (the mean over the
G's runs), W the expert set off their memory plans, and C the ruler's
`read_stream` and its pin rate. The shared arm reads W once and alpha x W for
each later M-tile, and no faster than C; 1 or above excludes nothing. The
ruler is the yaml calibrate wrote in this session, held to the bandwidth the
reports were scored against; after exfil it is the tracked yaml, and
`ruler=<yaml>` on `pairs-table` names another. Every rebuild prints each G's
bound with the ceilings it used. On session 5's pages, read on the laptop
with `ruler=` its published calibration yaml, it gives 0.841-0.842 at
read_stream and 0.886-0.888 at the pin rate at G = 4, 16 and 64, and above 1
at G=1, where a full re-read fits. `$SESSION/PAIRS-fixed.tsv` holds the
coordinates every row shares (model, tile, pinned config, treads, repeats,
duty) and where each was read, and the bound's inputs (the expert set, the
ruler, its two ceilings). `COUNTERS` and `COUNTERS.json` are the counter
probe's. Logs are under `$SESSION/chain-logs/`, and a V7 FAIL's follow-up
under `$SESSION/followup-g<G>.txt`.

**The price.** The laptop dry run of 2026-09-23 prices about 193 min of arms
(four R1 runs at 1124 s each at duty 1.0, 0.5, 0.25, which session 5's pod ran
in 1139-1160 s; twelve R3 runs at 590 s each at duty 0.25, the plan's 581 s
wall line plus the alignment probe's 9 s it leaves out), about a minute of
`tests/test_gpu.py` (38 tests at session 5's pod rate of 1.39 s a test), 8 min
of preconditions (the driver's own `arm_minutes`: thermal 3, calibrate 3,
pin_probe-n64-g1 2), 1 min for the counter probe (the same table's
`counter_plan`), 2 min for the probe check and 17 min of allowances (60 s of
compiles and weight build a ratio run, 5 min of exfil): about 222 min, about
$17 at $4.59/h, and it says book 5 h. No end suite is in that figure:
`END_SUITE=run` adds about 116 min (the tree's count, about 5000 tests, at
1.39 s a test) and the dry run then says about 339 min, about $26, book 7 h.
A V7 FAIL at seed 0 skips that G's two later seeds (about 22 min) and prints a
follow-up that costs about 77 min a G after the chain.

**Before releasing the pod.** The chain's calibrate re-dirties
`moe/bench/hardware/measured_nvidia_h200.yaml`, and every row measured after it
carries `git_dirty`: commit that yaml with the results. The exfil line printed
at the end carries the session directory, the results directory, that yaml,
calibrate's own run directory under `results/calibration/` and the console
`/workspace/alpha_g_chain.out`; copy it off before releasing the pod. A
follow-up run after the chain lands under the same session and results
directories, so the same line, run after it, carries that too.

## Before you rent anything

All of this runs on a laptop, costs nothing, and catches most of what would
otherwise be discovered on the meter.

```bash
bash scripts/pod_session.sh --dry-run          # every step, printed not run
.venv/bin/python -m pytest tests/ -q           # must be green; the count moves
# On the box the same command is P12 of pod_session.sh and runs from PY_BASE,
# the venv WITHOUT vLLM: `no_gpu` tests skip there (the -ra tail counts them),
# the gaps-session tests hide the card from every child they spawn, and a
# `--run`/bare invocation in a test is planted, never inherited from the box.
# Tests about committed artefacts read git's view (tests/_committed.py): the
# arms git TRACKS and the rulers git HOLDS, so calibrate's rewrite of the
# tracked ruler and a stray directory an earlier publish left under
# results/published fail nothing. Session 5's suite failed on the stray; the
# ruler tests sit in the half it never reached, and fail with its ruler
# planted into a laptop checkout of 33d2833.
# Never run the suite from the vllm venv: an unplanted --run would MEASURE.
bash scripts/run_all.sh --dry-run --profile crossing-uniform
.venv/bin/python scripts/alias_ablation.py --synthetic refit   # step 2b, no GPU
.venv/bin/python scripts/nsys_dram_probe.py --explain          # P-nsys, no GPU
```

The dry run degrades cleanly with no GPU and no vLLM: GPU checks report SKIP with
the reason, everything else runs for real. What it will NOT catch is anything
about the actual card, which is what the pre-flight is for.

**The test count is deliberately not written down here.** It was stale by a
hundred tests twice, because scripts land on this branch faster than the number
in a doc gets corrected, and a stale number teaches people to ignore the line.
P12 gates on `exit 0`, not on a count, and that is the thing to reproduce.

The last two lines are worth running before the pod because both exercise a whole
gate ladder off-GPU. `--synthetic refit` runs step 2b against a planted alpha and
must exit 0; `--synthetic retracted` plants 0.10 and must exit 1 with P1 FAIL,
which is what proves the gate can still refuse. `--explain` prints the sampling
arithmetic that decides what nsys can and cannot measure here, and it needs no
hardware at all.

Two things to do by hand before the pod exists, because both are slow to
discover late:

1. **Set `HF_TOKEN`, for download RATE rather than access.** Mixtral is no
   longer gated -- apache-2.0, `gated=False`, anonymous `config.json` download,
   verified 2026-09-01 -- so nothing here needs a licence accepted. But HF
   rate-limits anonymous transfers and step 0 pulls 93.4 GB against a 2:40
   deadline. `huggingface-cli` lives in the venv, not on PATH:
   `/workspace/venvs/base/bin/python -c "from huggingface_hub import login;
   from getpass import getpass; login(token=getpass())"`.
2. **Check volume size.** The sweeps download nothing at all -- they generate
   random weights -- but step 7 pulls 93.4 GB. 100 GB of Network Volume covers
   everything except trace capture; capturing Mixtral needs 250 GB.

---

## The pre-flight, and what each check kills

Runs automatically, takes about five minutes, spends almost nothing. Every check
below exists because something in this list actually went wrong on this project,
with one exception: P-nsys is not there to stop a failure, it is there because
what it finds changes what every later step is allowed to claim.

Run it alone with `bash scripts/pod_session.sh --preflight-only`.

| id | check | the class of failure it kills |
|---|---|---|
| P1 | working tree is clean | 76.7% of the canonical published pool carries `git_dirty = True`, and two whole arms are 100% dirty. Those rows are not reproducible from the commit they name. Waive with `--allow-dirty` and the waiver is recorded in the ledger. |
| P2a | GPU name matches `--expect-gpu` | `measured_<device>.yaml` resolves by NAME, so a second H200 pod silently inherits the first one's ceilings and an A100 would be scored against an H200 roof. **FATAL.** |
| P2b | at least 80 GB of device memory | deepseek-v3 at `E=256,N=2048` needs tens of GB for one layer. **FATAL.** |
| P2c | driver r580+ | a cu130 torch wheel needs it; on an older driver use the cu128 index. The index matches the DRIVER, not the image name. |
| P3 | torch 2.13.0 and Triton 3.7.1 | a different torch is a different CUTLASS, so the grouped GEMM you profile is not the one the published rows describe; a different Triton emits different PTX, so step 5 would answer about another compiler. |
| P4 | `override_config` binds and releases, on a shape vLLM has never seen | steps 2, 3 and 4 are all `override_config` experiments. If the hook does not bind they sweep nothing while printing a full table. deepseek-v3 (`E=256,N=2048`) is the sharpest probe available because vLLM v0.27.1 ships no tuned file for it on any card or dtype, so a bind failure cannot hide behind a file that happens to agree. **FATAL.** |
| P5 | an isolated `TRITON_CACHE_DIR` really produces PTX | with the shared `$WORKSPACE/triton-cache` inherited, every fused_moe specialisation is already built, nothing recompiles, no `.ptx` is written, and the dump script exits saying the kernel never compiled. This is very likely why the A100 was never successfully dumped. **FATAL.** |
| P6 | 110 GB on the volume, 10 GB on the container | the 93 GB download, and the several GB of temp space wheel extraction needs. |
| P7 | `entitled_ridge` still refuses 5 of the 14 published arms | the guard that stops an arm being quoted against another session's ruler. A change that silently stops refusing is invisible in any table. The count was "2 of the 10" until 2026-09-03; the three ladder arms published since carry no `measured.yaml` and are refused by construction, and `tests/test_docs.py` checks the number. The arms are the directories git TRACKS: an untracked one left on the pod's volume by an earlier publish is not counted. Session 5's checkout carried one (`2026-09-15-nvidia_h200-session3`, which that suite's census tests counted as an arm); P7 did not run there, and with it planted on a laptop the old directory listing reads it as a new refusal and FAILs. |
| P8 | the weights step 7 pulls are reachable | Asks whether the repos in `moe/spec.py` for `mixtral-8x7b` and `deepseek-v2-lite` resolve, using whatever credentials the box has. It used to check for a TOKEN and justify it with "Mixtral is gated" -- Mistral ungated that repo (apache-2.0, `gated=False`, `config.json` downloads anonymously), so the gate demanded a credential nothing needed and gave a reason that had stopped being true. A token still helps: HF rate-limits anonymous transfers and step 0 pulls 93.4 GB, so its absence is reported as an advisory rather than a failure. |
| P9 | the exact exfil paths are committable | an unanchored `plots/` rule matched at any depth and silently swallowed `results/published/<arm>/plots/*.png` on every publish. When this row was written zero `.png` files were tracked under `results/published/`; the rule is anchored now and 75 `.png` files are tracked (`git ls-files 'results/published/**/*.png'`, checked by `tests/test_docs.py`). **FATAL.** |
| P10 | which profiler exists | informational: whether `ncu` and `nsys` are on PATH. Whether `ncu` can READ a counter is the pod's to answer, not the platform's: two rented H200s refused with `ERR_NVGPUCTRPERM` (2026-08-25, `profiles/q2_kernel_names.txt`, and 2026-09-15), session 4's probe found no ncu on PATH, and the alpha(G) chain's counter probe asks on every pass (the chain's section above). `nsys` traces CUDA and usually works, but tracing kernels is not counting bytes and P-nsys below asks the harder question. |
| P11a | the step scripts exist and parse | several are written concurrently by other people. |
| P11b | those scripts accept the flags this session passes | a renamed flag should cost a line here, not an argparse error forty minutes in. |
| P11c | the suite interpreter carries no vLLM | soft: the tests plant every refusal door and hide the card from every child they spawn, so the suite runs from the venv WITHOUT vLLM (P12 does); an interpreter that imports vllm would let an unplanted bare invocation MEASURE, and the pod's own session 4 saw a `--run` go past its door |
| P12 | the test suite | a failure here costs seconds; the same failure found after an hour of benchmarking costs an hour, and every row in between is suspect. **FATAL.** |
| P13 | no active throttle, card under 60 C | thermal state is the largest source of run-to-run disagreement on rented hardware, and the harness records the symptom rather than controlling it. |
| P14 | the session directory is on a different mount from `/` | a Network Volume at `/workspace` survives termination; the container filesystem does not, and `/workspace` on a pod without a volume attached looks identical. **FATAL.** |
| P-nsys | can DRAM traffic be COUNTED here rather than modelled | not a failure gate. It decides whether steps 1 to 4 quote MEASURED or INFERRED bytes. **Never fatal, and absence is not even a soft FAIL.** Runs LAST in pre-flight because it is the only check that costs minutes. |

Pre-flight runs to the END even after a fatal, so one pass shows you every
problem rather than one per re-rent. It then stops before spending anything, and
reports the FIRST fatal, which is usually the cause of the rest. P-nsys is the
exception: once a fatal has tripped it is skipped, because that session is not
going to run and there is nothing for its answer to multiply.

### P-nsys, and why it moved to the front

**Every byte figure in this study is arithmetic.** Nothing here has ever counted a
DRAM transaction. No `ncu` counter has been read on a rented pod: two H200s
refused with `ERR_NVGPUCTRPERM` (2026-08-25 and 2026-09-15) and session 4's image
had no ncu on PATH, so compulsory-traffic bytes are computed from the shapes and
divided into a measured bandwidth. `nsys` reaches the DRAM counters by a different mechanism, sampling
rather than instrumenting, and whether it works on a given pod is an open
question this project has never answered.

**NOTHING IN THIS SECTION HAS TOUCHED A GPU, INCLUDING THE NUMBERS THAT LOOK LIKE
MEASUREMENTS.** The probe has never been run against a card. On a laptop it exits
3 with `REFUSED: no nsys on PATH`, which is what it should do and is not a
result. Three different kinds of statement live in this section and they are not
equally solid:

| statement | where it comes from | verified? |
|---|---|---|
| default 10 kHz, ceiling 200 kHz | NVIDIA's nsys documentation, hard-coded at `nsys_metrics.py:287` | **NO.** Never confirmed against any card, let alone an H200 SXM. |
| a 54 us launch buys 10.8 samples at that ceiling | arithmetic, `54e-6 * 200e3` | arithmetic is sound; it INHERITS the row above. |
| quantisation is charged per window, not per sample | arithmetic over the sampling model | same inheritance. |
| alpha to +/-0.156 at 10% traffic error | `alpha_uncertainty()`, propagated through the ladder | same, and it also assumes the sampler is unbiased, which is exactly what the calibration rung is for. |
| the sampler works on a rented pod at all | -- | **UNTESTED.** This is the probe's actual job. |

The direction of the risk is one-sided and worth stating: if the pod's real
ceiling is BELOW 200 kHz -- and the doc figure is a ceiling, so it can only be
lower -- then every conclusion above gets WORSE, not better. Fewer samples per
window, a larger edge term, a wider alpha band. A ceiling of 50 kHz would put a
54 us launch at 2.7 samples and widen the two-tile alpha band past the point
where it separates 0.558 from 0.10 at all. The probe MEASURES the rate actually
delivered rather than trusting the flag -- median inter-sample delta out of
`GPU_METRICS`, written to `$SESSION/nsys/probe.json` as `observed_sample_hz`
alongside a boolean `sample_rate_honoured` -- and that number, not the constant
in the source, is what any later claim must cite.

That check was added because the resolution arithmetic was self-referential
without it: `resolve()` divides the window by the period of the rate it
REQUESTED, so a build that clamped 200 kHz to 50 kHz would have left every
traffic total correct and every stated confidence wrong by a factor of four,
with nothing in the output looking odd. A clamp now sets `sample_rate_honoured`
false and forces `resolution_ok` false regardless of how the window scored;
`null` means the rate could not be measured at all, which voids the verdict
rather than widening it. The pre-check verdict is kept beside it as
`resolution_ok_before_rate_check` so the two causes stay distinguishable.

The one claim here that does NOT depend on the rate: **a single launch is
unmeasurable by orders of magnitude, not by a factor.** Reaching the minimum
samples per window for one 54 us kernel needs something above 370 kHz, so no
plausible correction to the doc figure rescues per-launch profiling. The merged
window design stands regardless of what the pod reports.

**It is a force multiplier, not a result**, and that is the whole reason it is in
pre-flight. If the sampler works, then steps 2, 3 and 4 and the dtype headline
stop being inferences from a byte model and become measurements checked against
counted bytes. Knowing that at 0:03 lets those steps say so. Knowing it at 3:00
is worth nothing.

**What it actually runs.** `scripts/nsys_dram_probe.py --calibrate`, capped at
twelve seconds per attempt. It asks the installed `nsys` which flags it offers
rather than guessing a spelling, walks a six-rung ladder that varies one thing per
rung, and scores each rung by whether `GPU_METRICS` came back non-empty with a
DRAM metric in it -- not by the exit code, because nsys exits 0 while writing a
report with no metrics in it. The last rung is a CONTROL with no metrics
requested, which separates "no sampler" from "no nsys". Then it profiles a
workload whose DRAM traffic is KNOWN without any model and checks the sampler
against it.

**Three outcomes, and only one of them is a FAIL.**

| outcome | what the session does |
|---|---|
| the sampler works and passes the known-traffic case | traffic is **MEASURED**. Step 2 gets a companion `--measure` run, and every later step says MEASURED. |
| no invocation sampled a DRAM metric | traffic is **INFERRED**. This is the EXPECTED outcome on a rented pod, it is an INFO row and not a FAIL, and the session runs exactly as it did before. |
| the sampler works and gets the known answer WRONG | a soft **FAIL** (`PnsysCal`). This is the one outcome worth a verdict, because an instrument that answers and answers wrongly would have corrupted every step after it silently. Traffic stays INFERRED. |

**What it does NOT buy, and this matters.** Even a working sampler cannot measure
a single MoE kernel launch. The rate ceiling nsys offers is 200 kHz, so one 54 us
launch buys ten samples and the quantisation is charged per WINDOW rather than per
sample: 10 ms of kernel time is usable as one contiguous window and useless as ten
separate ones. Profiling a thousand separate launches buys samples and zero
accuracy. So wrapping a 45-minute sweep in nsys would be pointless, and the
session does not do it. What IS measurable is a long contiguous run of
back-to-back launches merged into one window, which is why the traffic
measurement is a companion run of the cell rather than a trace over the sweep.

At 10% traffic error the sampler pins alpha to about +/-0.156 on a two-tile cell.
That **discriminates 0.558 from the retracted 0.10 and cannot pin either**;
+/-0.05 needs 3.2%. More M-tiles per expert sharpen it, which is why the cell
profiled is a deep one.

**Skip it** with `--no-nsys`. The session then runs as it did before P-nsys
existed and everything says INFERRED.

---

## The session

| offset | step | script | what it costs |
|---|---|---|---|
| 0:00 | 0. mixtral weights, backgrounded | `huggingface_hub.snapshot_download` | nothing on the critical path |
| 0:00 | pre-flight, P-nsys included | `scripts/nsys_dram_probe.py --calibrate` | about 5 min |
| 0:05 | 1. fp8 same-session calibration | `scripts/calibrate_hardware.py` | about 3 min |
| 0:08 | 2. BLOCK_M sweep, multi-tile | `scripts/block_m_crossing_sweep.py` | 4-6 min |
| 0:14 | 2. traffic measurement of the cell | `scripts/nsys_dram_probe.py --measure` | 2 min, only if P-nsys said yes |
| 0:16 | **2b. alias ablation, the independent alpha** | `scripts/alias_ablation.py --run` | 6-10 min |
| 0:26 | 3. GROUP_SIZE_M sweep | `scripts/group_m_alpha_sweep.py` | 12-15 min |
| 0:41 | 4. tuned vs forced fallback | `scripts/tuned_vs_fallback.py` | capped at 35 min |
| 1:16 | 5. config and ISA provenance, PTX | `scripts/check_mma_path.sh` x 3 cells | 15-30 min |
| 1:46 | 6. dense uniform grid | `scripts/run_all.sh --profile crossing-uniform` | about 54 min |
| 2:40 | 7. trace capture | `scripts/capture_traces.py` | 25-40 min |
| 3:20 | 8. exfil, and the two alphas reconciled | `scripts/publish_results.sh` plus checks | about 10 min |

**Does it still fit? Yes, with more room than the old plan had.** Worst case on
the pessimistic arm of every range reaches exfil at 3:20 and finishes at 3:30,
against 4:20 and 4:30 before; best case is 2:51. Two things bought that back: the
BLOCK_M sweep and the GROUP_SIZE_M sweep are both quicker than the original
estimates, and the two additions are small. The 93 GB download needs about 90
minutes and step 7 does not start until 2:40, so it is still nowhere near the
critical path -- which was the reason step 0 exists.

**Step 2b runs immediately after step 2 on purpose.** Both estimate alpha. One
session and one thermal state means a disagreement between them cannot be
explained away by the card having been a different card.

Resume anywhere: `--from 5`, or `--only 6`. `--only` takes any step id including
`2b`; `--from` takes a number, and `--from 2` includes 2b because 2b sits at
position 2 in the running order. A step that already has a PASS and no FAIL in
`$SESSION/LEDGER.tsv` is skipped; `--force` re-runs it. The sweeps resume through
the harness's own `--run-id` manifest, which flushes per cell, so aborting costs
at most one cell.

---

## Step 0 (0:00) -- mixtral weights

**Why first.** 93.4 GB at roughly 90 minutes. Step 7 needs it at 3:40. Starting it
at 0:00 is the difference between a 4 hour 30 session and a 6 hour one.

**PASS** the background process started. **FAIL** nothing started, and step 7 will
say so again with the log tail.

**If it is still downloading at 3:40**, do not kill the pod. The volume keeps the
partial download and `snapshot_download` resumes. Capture `deepseek-v2-lite`
(31.4 GB, ungated) in the meantime.

---

## Step 1 (0:05) -- the fp8 same-session calibration

**One line, and it gates two results.**

Two published statements currently rest on a calibration that did not belong to
the session that measured them:

1. the dtype headline (bf16 1.162 against fp8 1.361 pooled, 1.475 matched) comes
   from an arm `entitled_ridge` REFUSES, because that arm's calibration measured
   no fp8 ceiling;
2. every fp8 row in it carries `achieved_peak_tflops = 0.0`, so its
   `implied_traffic_ratio` column is empty and `alpha` cannot be fit across
   dtypes at all.

**What this step does and does not fix.** Measuring an fp8 ceiling here makes
every fp8 row measured TODAY quotable. It does NOT rescue the published fp8 arm.
Restamping those old rows against today's ruler would be exactly the borrowed
calibration of `docs/INSTRUMENTATION.md` defect 7, and `recompute_ceilings.py` is
the wrong tool for it. The published arm stays refused until its rows are
re-measured.

**Prediction**, from the four H200 calibrations committed on this branch
(2026-09-02, 09-09, 09-10, 09-21; a fifth, 152.9 on 2026-09-14, sits on
`pod-h200-session3` unadopted and is superseded by the 09-21 file):

| quantity | expect | note |
|---|---|---|
| bandwidth, triad | 4374-4378 GB/s | reproduces to 0.09% across seven calibrations |
| dense bf16 | 663-771 TFLOP/s | the term that does NOT reproduce: 16% between the extremes of seven calibrations (712.4, 701.6, 770.9 on 2026-08-28; 712.3, 668.5, 682.1, 663.0 since) |
| bf16 ridge | about 151.4 FLOP/byte, band 143.7-151.4 (read_stream..triad, `calibrate.py`'s band; the sweeps' all-pattern band the rescored reports carry is 141.6-154.1) | the card's 2026-09-21 calibration, 663.0 TFLOP/s at 1455 MHz over 4378.0 GB/s. This row said 155.9 from the 2026-09-10 calibration until 2026-09-21, 152.8 from the 2026-09-09 calibration until 2026-09-10, 162.8 from the 2026-09-02 one until 2026-09-09, and before that a "band every absolute figure carries" of 160.3-176.2 until 2026-09-02 (retracted: that spread is the compute ceiling failing to reproduce, not any card's own band, `docs/FINDINGS.md` RETRACTIONS (e)). Read the committed `moe/bench/hardware/measured_<card>.yaml` rather than this row: five rentals have moved it and the between-rental spread in the compute ceiling is the largest single uncertainty in any roof fraction quoted here |
| fp8_e4m3 | about 1437 TFLOP/s | 2.17x the bf16 figure, at a 1395 MHz median on 2026-09-21 whose clock fell 1395 -> 1320 during the GEMM: DRIFT FAIL by the calibrator's own first-to-last rule, printed on the page and scored by no gate (`fp8_gemm_clock` is not in `UNDER_LOAD_BLOCKS`), so the ceiling is published as measured and its only consumer, the dtype arm, recalibrates on its own pod; 1454 at 1380 MHz on 2026-09-10 |

**The gates.**

| gate | meaning of a FAIL |
|---|---|
| S1a exit 0 | **STOP.** Without a calibration the sweep runs with EMPTY efficiency columns. On one occasion an H100 pod silently satisfied the repo's committed H200 yaml and the whole sweep produced blank efficiency. |
| S1b `checked_on` is today | **STOP.** The yaml was not rewritten, so this session would publish against another session's ruler. That is defect 7 exactly, and it cost claim C5 its target for three days. |
| S1c fp8 peak > 0 | any fp8 row measured today is again unquotable. **On an A100 this SHOULD fail**: Ampere has no fp8 tensor cores and a number there would be fiction. |
| S1d bandwidth 3900-4800 | above the band means your buffer fit in cache and it is not a DRAM measurement; below means the card is contended. Every `implied_traffic_ratio` is divided by this. |
| S1e ridge 150-185 | every AI-cap prediction in this session was stated against 160.3 (retracted 2026-09-02: 160.3 is one earlier calibration, not this card's ridge, and the cap identity itself is retracted; the predictions are re-derived through `ai_model.cap_from_fitted` at THIS session's published ridge, as brackets over `alpha_a`). Outside the band, re-derive them before reading step 2. |
| S1f fp8 rows carry a peak | the yaml has the ceiling but the driver is not stamping it onto rows, which repeats the defect this step exists to close. Look at `moe/bench/driver.py` around `achieved_peak_tflops`. |

The step also snapshots the yaml and its sha256 into `$SESSION/calibration/`.
Step 8 checks that sum again. That guard exists because on 2026-08-28 a
recalibration overwrote `measured_<device>.yaml` between a sweep finishing at
19:21 and its publish at 19:35; the two rulers disagree by 9.9% on the compute
ceiling and nothing downstream could tell.

---

## Step 2 (0:08) -- the BLOCK_M sweep, and the session's central result

**The science.** Arithmetic intensity of an MoE expert GEMM is

```
AI(r) = (2r/b) / Q(r),    Q(r) = 1 + alpha (ceil(r/BM) - 1)
```

The first M-tile reads the expert's weights in full and each additional M-tile
re-reads them, discounted by L2 by a factor `alpha`. Two consequences. AI is
BOUNDED at `2 BM / (alpha b)` (retracted 2026-09-02 for a FITTED alpha: the
study's alpha is a ladder fit, and the cap it implies is
`2 BM / (alpha b) / (1 + phi + delta)`, a bracket over the unmeasured
`alpha_a`; `moe/bench/ai_model.py`, `docs/FINDINGS.md` RETRACTIONS (a)), so a
tile height can put the compute roof permanently out of reach. And the crossing solves `R = ridge b Q(R) / 2`, which is
a step function on both sides and can therefore have several solutions or none.

**alpha was refit on 2026-09-01 from 0.10 to 0.558.** 90% band 0.529-0.588 over
10,813 admitted rows, placebo -0.002. The 0.10 was an estimator artefact: it came
from minimising the CV of a POOLED ratio, an objective that falls 0.7% across its
whole range and lets alpha absorb a between-cell level trend running the wrong
way. Changing only the estimator on the original 151 rows gives 0.484.

**Prediction, ridge 160.3, bf16.** (Retracted 2026-09-02 as written: the caps
read a fitted alpha into the `alpha_b` slot and 160.3 is one earlier
calibration, not this card's ridge, `docs/FINDINGS.md` RETRACTIONS (a), (b),
(e). Through (EXA) at alpha = 0.558 the 128 cap is 214.1 at `alpha_a = 0` and
169.1 at 0.143, crosses the H200's 162.8 only for `alpha_a < 0.17`, and the
two measured 128 ladders give 130.7 and 135.4, below both cards' ridges. The
rows stand as the registered prediction of the 2026-09-01 session.)

| BLOCK_M | AI cap | crossing | mixtral | qwen2 | deepseek-v3 |
|---:|---:|---|---:|---:|---:|
| 32 | 57 | **none, at any batch** | -- | -- | -- |
| 64 | 115 | **none, at any batch** | -- | -- | -- |
| 128 | 229 | R = 250 | 999 tok | 1998 | 7992 |
| 256 | 459 | R = 160 | 641 tok | 1282 | 5130 |

128 and 256 must separate by **1.56x**. At the retracted alpha = 0.10 all four
crossed and the spread was 1.10x. The two alphas are QUALITATIVELY different here,
which is what makes this worth a pod.

**Run it in the multi-tile regime.** C3 measured the tile at T=16, where every
expert is one tile at every BLOCK_M, so there were no re-reads to save and only
occupancy could move. That regime is not this one and its null result does not
transfer. Wave count must exceed about 10 on BOTH sides of a step, so occupancy is
saturated and any remaining movement is traffic.

**What PASS means for the paper.** The tile-corrected roofline survives a test
that could have killed it, and the ceiling `2 BM / (alpha b)` becomes a measured
result rather than a proposal. It also makes a strong statement about production:
vLLM's tuned configs run BLOCK_M = 16 through the whole decode range, and at
alpha = 0.558 that caps AI at 29 against a ridge of 160.3, so a decode-configured
MoE kernel is structurally incapable of reaching its compute roof at any batch
size. (Qualified 2026-09-02: the 29 is a (LIN) point and the corrected cap is a
bracket, 28.4 at `alpha_a = 0` to 22.8 at 1, so the conclusion at BLOCK_M=16
survives; but on uniform routing vLLM runs 16 multi-tile in 1 of 24 cells, so
the cap binds almost nowhere and `scripts/tile_cap_test.py` is demoted to a
test of the formula, `docs/FINDINGS.md` RETRACTIONS (i).)

**What FAIL means, and there are two different ones.**

- **32 or 64 DOES cross.** Then `alpha < 0.0998` after all (retracted 2026-09-02
  as a threshold: 0.0998 is `BM/ridge`, the (LIN) identity; the corrected
  threshold goes through `ai_model.cap_from_fitted` and is a bracket over
  `alpha_a`), the ceiling is real but higher than the refit says, and the refit
  needs redoing. This is the more interesting failure and it is worth wanting.
- **128 and 256 separate by about 1.10x rather than 1.56x.** Then the uncorrected
  `2R/b` describes the data and the whole tile-corrected section retracts.

Neither is a broken run. Both are results. Do not retry either.

**Confound, named rather than hidden.** BLOCK_SIZE_M sizes the register
accumulator, so changing it changes occupancy as well as traffic, and a time
change is ambiguous between the two. That is why the sweep is designed around
where the crossing sits rather than around raw time.

**If the step dies.** Steps 3 and 4 still stand: they measure how alpha VARIES,
which is a separate claim from its level.

**Read it with step 2b, not alone.** This step tests a consequence of alpha
against a byte model. Step 2b measures alpha without one. A disagreement between
them is not noise, and this step on its own cannot tell a wrong alpha from a wrong
byte model.

---

## Step 2b (0:16) -- the alias ablation, and the second alpha

**Why a second estimate at all.** `alpha = 0.558` carries the whole
tile-corrected roofline, and it comes from ONE regression against a byte model
with no tile term in it: `alpha_refit` fits it out of `implied_traffic_ratio`,
which is `time x bandwidth / COMPULSORY BYTES`. C4 in `docs/FINDINGS.md` is a
CONFIRMED finding that the compulsory-byte ruler was itself wrong by 1.85% until
it was fixed. A number a paper's headline rests on should not rest on one
estimator over one derived column.

**The method, which touches none of that.** Run the same grouped-GEMM access
pattern twice at `n = 1, 2, 4, 8` M-tiles per expert.

- **NORMAL** reads each expert weight block once per M-tile, so its HBM weight
  traffic is `W (1 + alpha (n-1))` in units of one full pass.
- **ALIASED** points every weight load at ONE resident tile. The loads still
  execute, the instruction stream is identical, but they HIT L2 instead of
  missing to HBM.

Everything else -- activation reads, output writes, arithmetic, launch, grid --
is identical and cancels:

```
D(n) = T_normal(n) - T_aliased(n) = W (1 + alpha (n-1)),   D(1) = W
alpha = (D(n)/D(1) - 1) / (n - 1)
```

`W` is never predicted from bytes and a bandwidth. It is MEASURED, in the same
units, in the same session, by the same clock. Rows per expert is an exact
multiple of `BLOCK_M` at every rung, so the padding term is exactly zero and the
tile count is the only thing that moves.

**Why the answer is an interval and not a point.** The aliased variant issues the
same loads; what it does not do is MISS. So both variants push `n W` bytes through
L2, and that common cost cancels only if L2 and HBM service ADD. A streaming
kernel runs closer to `max(L2, HBM)`, where the naive difference estimator fits to
`(alpha - r)/(1 - r)` and, at an `r` near 0.5 on this card, a true 0.558 would
come back as **0.018** -- which is to say the naive estimator is biased toward
almost exactly the value this repo retracted. So a second estimator, exact under
`max` and biased the other way under addition, is run beside it and the answer is
the BRACKET between them. The report prints the measured `r`, which is what sets
the bracket width.

**PREDICTION.** The interval contains 0.558 and excludes 0.10 and TEMPO's 0.33.

**The gates**, all of which appear in the log as `[PASS]` or `[FAIL]` lines that
`pod_session.sh` counts.

| gate | meaning of a FAIL |
|---|---|
| ISA | the two variants did not compile to the same instruction stream, so the difference is not measuring what it claims. **Nothing from the run may be quoted whatever P1 says.** The gate counts the UNION of `ld.global` and `cp.async`, because Triton pipelines the loads at `num_stages > 1` and counting `ld.global` alone would read zero. |
| correctness | a variant did not reproduce its closed form, so the aliasing scalars did not do what was intended. |
| placebo | two identical launches differ by a large fraction of D. The design is measuring noise. |
| signal | the weight read is not most of what the kernel does, so the difference is measuring something the label does not cover. |
| form | `D(n)` is not affine in `(n-1)`, so slope-over-intercept is not alpha. |
| control | an L2-resident geometry, whose per-expert block fits in L2 and therefore has no HBM re-read to save, showed an extra-tile cost anyway. Whatever it is, it is not weight traffic. |
| resolution | the interval is too wide to separate the candidates. The run says **NOT TESTABLE** rather than picking one. |
| P1 | the interval does not contain the refit. **This is the refutation and it is a result.** The gate line names which candidate the interval DOES contain. |

**Exit codes are meanings, not error levels.** `0` every gate passed. `1` a gate
failed, which is a refutation. `3` it could not run here; the deepseek-v3 rung
needs about 16 GiB free on the card. `4` the design did not identify alpha and the
honest answer is NOT TESTABLE.

**What this is NOT.** It is not vLLM's `fused_moe_kernel`. It has vLLM's B-pointer
arithmetic, vLLM's `GROUP_SIZE_M` swizzle, vLLM's `[E, N, K]` layout and vLLM's
tile constants, with the reduction replaced by `acc += tl.sum(a) + tl.sum(b)` on
`docs/STUDY.md`'s own instruction. That replacement is what keeps the estimator
unbiased: with a real `tl.dot` the aliased variant goes compute-bound while the
normal one stays memory-bound, `D(n)` loses a copy of the per-tile compute cost
and alpha is biased DOWN. `--compute dot` runs it that way anyway, prints the
bound, and refuses to answer P1 with it.

**One residual confound, bounded rather than removed.** Aliasing to a single tile
pins every load to a handful of L2 slices, so the aliased ladder is served more
slowly than aggregate L2 bandwidth suggests. That inflates it, and the effect is
contained by the bracket rather than eliminated. Spreading the alias would remove
the closed form the correctness gate checks against, which is a worse trade.

**Rehearse it off-GPU** with `--synthetic refit` (must exit 0) and
`--synthetic retracted` (must exit 1 with P1 FAIL). `--replay <dir>` re-reports a
finished run without a GPU.

---

## Step 3 (0:26) -- GROUP_SIZE_M, is alpha a scalar

**What it tests.** The refit found alpha falling with GROUP_SIZE_M: 0.570 at 1,
0.488 at 16. That is exactly what a swizzle-for-L2-reuse mechanism predicts, which
turns alpha from a fudge factor into something with a named cause. But
**GROUP_SIZE_M 32 and 64 have ZERO discriminating rows in the published pool**, so
the direction is untested beyond 16 and cannot be tested from existing data at any
effort. Only override_config varying it settles it, and NOT at fixed batch:
group_m_alpha_sweep.py deliberately refuses that instruction, because one batch
cannot identify alpha under this estimator -- the token count IS the intercept,
so a single x-level is absorbed exactly and only curvature is left. On the
design's own x values at the median measured spread of 0.77% the top rung alone
gives a 90% band of 0.395-2.310 against the seven-rung ladder's 0.550-0.593, 44x
narrower against an effect size of 0.082 (the top rung is 384 since 2026-09-21,
when the fourth H200 calibration put 448's worst routing realisation over the
preflight's 90%-of-the-band line). The ladder is identical across every GROUP_SIZE_M, so the
cross-setting comparison is still at fixed design.

**Prediction.** alpha keeps falling monotonically at 32. (Retracted 2026-09-02 as
anything more than a registered prediction, `docs/FINDINGS.md` RETRACTIONS
(d): on the published surfaces the apparent direction of alpha with
GROUP_SIZE_M, on either card, was a pooled artefact, the G=1 and G=64 medians
being different models; the one matched A100 cell moves the other way by
0.028 and a paired MDE needs two cells. No direction is established, so a
rise here is not a "FAIL against a finding", it is the first measurement.)
**g=64 is NOT answerable on the
mixtral arm this step runs**: num_pid_m tops out at 59-61, so g=64 saturates at
every rung and the setting cannot be distinguished from g=32. Answering it needs a
second arm, `--model qwen2-57b-a14b --tokens 32,64,128,256,512,768,1024`, which
`pod_session.sh` does NOT invoke. Run it by hand if step 3 holds at 32 and the
trend matters, and the fall
flattens as the swizzle stops buying reuse.

**PASS** means alpha may be reported with a mechanism instead of as an
unexplained constant. **FAIL**, meaning a rise at some point, refutes the swizzle
story and alpha goes back to being a fitted number. A non-monotonic alpha is a
real result: report it, do not retry it.

Note either way that alpha is not a scalar. It already drifts with BLOCK_M, 0.466
at 64 and 0.625 at 128, so any single number carries a range.

**This step is also the session's REGRESSION estimate of alpha.** Its
`GROUP_SIZE_M = 1` row is what step 2b's ablation is reconciled against in the
summary, and unlike the published refit it was measured on this card in this
session. If step 3 does not run, the reconciliation falls back to the published
0.558 / 0.529-0.588 and says in the file that the number is not from this card.

---

## Step 4 (0:41) -- tuned config against a forced fallback

**What it tests.** Only 2 of the 8 (model x card) cells in this study have a tuned
vLLM config at all, both on the H200: `E=8,N=14336` (mixtral) and `E=64,N=2560`
(qwen2). The other six take the hardcoded bf16 ladder, `M<=32 -> 16`,
`M<=96 -> 32`, `M<=512 -> 64`, else 128, and vLLM says so on the log with "Using
default MoE config. Performance might be sub-optimal!". Nothing in this study has
ever measured what that warning is worth.

**Prediction.** In the memory-bound regime the difference is small, because the
tile is not on the critical path there. In the multi-tile regime the tuned file
climbs to BLOCK_M = 128 at M = 256 while the fallback sits at 64, and the ceiling
says only 128 can ever cross (retracted 2026-09-02 as a point: whether 128
crosses is a bracket over `alpha_a`, and the two measured 128 ladders sit
below both ridges, `docs/FINDINGS.md` RETRACTIONS (b)). So the gap should OPEN
with batch rather than being a constant offset.

**Why it matters beyond the number.** This repo once published "BLOCK_M is not a
knob" and had to retract it, because vLLM and SGLang both ship tuned
`BLOCK_SIZE_M`. The honest question is what the tuning buys, not whether it
exists, and this step is the answer.

**PASS** gives a number for the cost of running six of eight cells on a fallback
ladder. **FAIL** as "no difference at any batch" is publishable too and says the
tuned files buy nothing in this regime.

---

## Step 5 (1:16) -- config and ISA provenance, and the file that must leave the pod

**Why this step is really about exfil.** C1 and C3 are the only claims in
`docs/FINDINGS.md` that rest on transient pod output. The PTX dumps, the CUTLASS
kernel names and the "Using default MoE config" warning were all quoted from run
logs that were never committed. Every other claim in that file can be recomputed
from `results/published/` on a laptop; those two cannot be checked without a GPU,
which is the hole a reviewer opens first.

**Three H200 cells, three different predictions**, derived by `tile_resolve`
against the vLLM v0.27.1 config snapshot:

| cell | resolved tile | ISA prediction | config line |
|---|---|---|---|
| deepseek-v3 T=16 | BM=16, warps=4 | wgmma **= 0**, mma.sync > 0 | "Using default MoE config" |
| deepseek-v3 T=256 | BM=64, warps=8 | wgmma **> 0** | "Using default MoE config" |
| mixtral T=256 | BM=128, BN=256, warps=8 | wgmma **> 0** | "Using configuration from" |

Triton selects Hopper's `wgmma.mma_async.m64nNk16` only when
`BLOCK_M % 64 == 0` AND `num_warps % 4 == 0`. So the first cell declining it and
the other two reaching it is ONE prediction, not three. The third is the tuned
specialisation FINDINGS names as never having been compiled to disk.

**On the A100 the prediction is different and simpler.** Below compute capability
9.0 `getMMAVersionSafe` returns `{2}` alone, so NO tile reaches the warpgroup
instruction at any size. Both cards means two pods: run `--only 5` again on an
A100 session and exfil that census too.

**PASS on all three** turns "the decode path is on the Ampere-era instruction, on
Hopper silicon" from an inference into a measurement, with a committed file behind
it.

**FAIL, and what each one means.**

- **wgmma present at T=16.** C3 is REFUTED. Triton is reaching M=64 some other way
  and the inference from `BLOCK_SIZE_M = 16` was wrong.
- **No wgmma at BM=128.** Triton declines the warpgroup instruction even when the
  tile allows it, which is a finding in its own right and a bigger one than C3.
- **config line "unlogged".** vLLM emits it once per `(E, N, dtype, device)` via
  `info_once`, so a second cell in the same process is silent. Check this was the
  first `fused_experts` call and that the log level lets info through. Without the
  line, every tile statement about the ten published v3 arms stays DERIVED from
  vLLM's source rather than OBSERVED.

**Each cell gets its own dump directory.** `check_mma_path.sh` begins with
`rm -rf` on its `--out`, so three cells sharing one directory would leave only the
last one's PTX.

**The dumps leave as a tarball.** `*.ptx`, `*.so`, `*.nsys-rep` and `*.qdrep` are
gitignored at any depth on purpose, so raw dumps cannot be committed. The step
writes `$SESSION/exfil/ptx-<card>.tar.gz` and a readable
`$SESSION/exfil/ISA_CENSUS.txt`, and pre-flight P9 has already proved both
filenames are committable.

---

## Step 6 (1:46) -- the dense uniform grid

**The profile.** `crossing-uniform`: uniform routing only, 7 seeds, a `2^(1/4)`
grid from 1 to 16384 straddling the ridge band, L2-WARM eager.

Three things in that sentence are decisions, not defaults.

- **Uniform only.** `2R/b` is a uniform-routing statement and pooling the seven
  routing regimes is INVALID for a crossing, not merely noisy: under skew the busy
  experts are compute-bound while the quiet ones are still memory-bound at the same
  batch, so the layer straddles the ridge and there is no single crossing. Dropping
  `--routing uniform` from any crossing command changes the numbers by up to 4.3x.
- **L2-warm.** The cold basis loses 5 of 8 one-stage crossings to throttle
  exclusion (by the retired idle-instant flag, which detected the idle-boost
  catch rather than throttling, `docs/FINDINGS.md` RETRACTIONS (f); LEVEL and
  DRIFT replace it on the instrument, and only DRIFT excludes a row since
  2026-09-09).
- **Read it with `octave_ladders`.** Fed whole to `crossing_from_points` this grid
  is biased 4-18% LOW and twice as wide as the powers-of-two grid it extends.

**Why the tile should be pinned.** Along an unpinned grid the tile CHANGES with
the token count -- mixtral climbs 16, 32, 64, 128 across the sweep -- so the
staircase the detector reads is partly the config ladder stepping and partly the
roofline transition, and `crossing_from_points` returns the FIRST crossing, which
is usually a tile step. That is instrument defect 3.

The step probes for a pinning hook by running one cell with
`MOE_FORCE_TILE={"BLOCK_SIZE_M":128,...}` and reading the OBSERVED `tile_block_m`
column back out of the CSV, because "the env var was set" and "the kernel ran that
tile" are different facts. **If gate S6a fails, nothing in the sweep path honours
the hook**, the grid runs on vLLM's own ladder, and its crossing may NOT be
described as tile-pinned. Use step 2 for the pinned answer instead. The unpinned
grid is still worth having: it is the crossing under the configuration that
actually ships.

**Prediction** at a pinned BLOCK_M = 128: mixtral crosses near 999 tokens, qwen2
near 1998, deepseek-v3 near 7992. At BLOCK_M = 64 the cap is 115 and no crossing
exists anywhere on the grid. (Retracted 2026-09-02 as points: 999 / 1998 / 7992
solve the (LIN) cap at ridge 160.3, and 115 is the (LIN) cap; corrected, the
64 cap is 110.7 at `alpha_a = 0` and lower above it, so "no crossing" at 64
survives, while 128's crossing is a bracket over `alpha_a`, `docs/FINDINGS.md`
RETRACTIONS (a), (b), (e).)

**The gates.**

| gate | meaning of a FAIL |
|---|---|
| S6a tile pinning honoured | the grid is unpinned. See above. Not fatal. |
| S6b sweep exit 0 | resume with `--from 6` and the run id the step printed. |
| S6c zero correctness failures | **STOP.** The kernel computed the wrong layer, so every timing in the arm is a timing of the wrong thing. Do not publish it. |
| S6d under 5% DRIFT | The count is the driver's `throttled` column (`moe/bench/driver.py`), which SINCE 2026-09-09 is `clock_drift_ok is False` and nothing else. A drifted cell never reached one operating point, so its timing is of two clock states averaged together; those are the rows the rule excludes from crossing detection, and a high rate narrows the grid the detector can actually use. Fix it at the instrument (warm until two consecutive clock reads agree within one 15 MHz step) rather than by widening the gate: all 135 drifted cells of the 2026-09-09 session sit at the first cell of a rep after a workload change, the governor settling. FOUR of them carry samples on disk and show it directly (1875 to 1725, 1560 to 1650, 1560 to 1650, 1620 to 1710); the other 131 are inferred from their position in the rep, because pre-v7 rows kept only the first and last sample. `docs/APPARATUS.md` section 1 states the same split. A LEVEL failure on EITHER side (`clock_level_side` = `low` or `high`, written on every row that fails it) is NOT throttled and NOT an exclusion anywhere. Under the 700 W cap the clock under load is set per tile by the kernel's own draw, so `high` is a memory-shaped cell boosting (the calibration's memory load holds 1980 MHz) and `low` is a hungry tile at its steady state (BLOCK_M=128 holds a median 1395 MHz over 196 cells against a 1485 MHz GEMM reference); the per-tile table is in `docs/APPARATUS.md` section 1. What is wrong for both is the fixed-roof fraction, and `pct_of_roof_at_cell_clock` is the column to read beside it. A gate or ladder that excludes on `clock_level_ok = failed` carries one of the two readings this repo has already paid for: the one-sided one found at thirteen consumers on 2026-09-08, which counts every memory-bound cell as throttled, or the two-sided one that landed the roofline and depth arms INVALID on 2026-09-09 by excluding 15 of 39 cells and 110 of 168 treads that were never throttled. Read either as a defect in the gate, not as a throttle rate, and fix the gate before resuming; the proof a gate is correct is a planted 1980-against-1485 row with side `high` and a planted 1395-against-1485 row with side `low` that it both KEEPS, and a planted DRIFT row that it excludes. (Before 2026-09-02 this gate counted the retired idle-instant flag, which fired on 91% of vLLM rows above T=4096 on the alpha-0558 arm while flagged and unflagged replicates timed at ratio 0.998.) |
| S6e coverage against the planner's own row count | the sweep stopped short, almost certainly on `--max-minutes`. Resume rather than reading a crossing off a truncated grid. |

**Two defaults that are not measurements, and that any new analysis of these rows
must filter.** `ms_p50 = 0.0` means the cell never ran, not that it took no time;
`crossing.timed_rows` drops those. `implied_traffic_ratio = 0.0` means the column
does not apply, because `driver.py` writes it only for memory-bound cells; nothing
guards that one and the filter has to be written per analysis.

---

## Step 7 (2:40) -- trace capture

**What it fixes.** `traces/` holds a single `.gitkeep`. Every routing distribution
in this study is parametric, so every claim about realistic skew rests on zipf, hot
and dirichlet standing in for measurements never taken, and `capture_traces.py`
has never been run. Captures are kilobytes and are committed on purpose; model
weights never enter the repo.

**What fits on one H200 (141 GB, bf16).**

| model | weights | capturable here |
|---|---:|---|
| mixtral-8x7b | 93.4 GB | yes, comfortably |
| qwen2-57b-a14b | 114.8 GB | yes, about 26 GB left for KV and activations |
| deepseek-v2-lite | 31.4 GB | yes, the cheap 64-expert proxy |
| deepseek-v3 | 1369 GB | **no**, needs 5+ cards |

DeepSeek-V3's routing cannot be captured on this hardware. Benchmark its GEOMETRY
with parametric routing and say so explicitly wherever the result appears.
Claiming a captured V3 trace would be false and is the kind of thing a reviewer
checks first.

**FAIL** for a model means its skew claims stay parametric. That is a visible gap
rather than a silent one, because `traces/` is tracked.

---

## Step 8 (3:20) -- exfil, and nothing is torn down before it passes

**This repo has lost work twice at exactly this point.** Every published figure
was dropped by an unanchored `plots/` rule that `git add` applied silently, and
the A100 PTX was never produced because a shared Triton cache meant nothing
recompiled. Both losses were invisible at the terminal.

**What must leave the pod.**

1. the sweep CSVs and manifests, via `publish_results.sh`, with the calibration
   they were measured against beside them
2. that calibration's sha256, unchanged since step 1 wrote it
3. `ISA_CENSUS.txt` and `ptx-<card>.tar.gz` from step 5
4. `traces/*.npz` from step 7
5. every step log, so a number can be traced to the run that made it
6. `LEDGER.tsv`, the machine-readable record of every gate above
7. **`$SESSION/alias_ablation/`**, step 2b's report and per-cell measurements.
   This is the only estimate of alpha in the session that does not go through the
   byte model, so it is the only thing that can corroborate or refute the number
   everything else is quoted against.
8. **`TRAFFIC_PROVENANCE.txt`** and `$SESSION/nsys/probe.json`, which say whether
   any byte figure here was counted or modelled.
9. **`ALPHA_RECONCILIATION.txt`**, the two estimates side by side with their
   intervals and the verdict on whether they agree.

Items 7, 8 and 9 were added on 2026-09-01. The manifest had already omitted the
outputs of steps 2, 3 and 4 once -- the entire scientific payload -- and a pod
torn down after a clean run would have taken the answer with it. Items 8 and 9 are
listed UNCONDITIONALLY, unlike everything above them, because they are written by
the session rather than by a step and their whole point is to exist even when a
step did not run: "traffic was INFERRED because nsys is not installed" and "only
one alpha exists" are both results that have to survive teardown.

**The two alphas are reconciled here, before the manifest is built.** Step 8
starts by reading step 3's regression alpha and step 2b's ablation interval out of
their logs, writing both to `ALPHA_RECONCILIATION.txt`, and saying plainly whether
they agree. The summary then prints that file ABOVE the gate counts, so it is the
last thing on the screen. A session that measured the study's central parameter
twice and buried the disagreement four hundred lines up a log would be a session
that answered the question and told nobody.

**Read the intervals with their asymmetry.** The regression band is a cluster
bootstrap over sampling noise. The ablation interval is a BRACKET between two
estimators biased in opposite directions by whether L2 and HBM service add or
compose as a max. They are not the same kind of object, so overlap is weak
evidence of agreement and disjoint is strong evidence of disagreement.

**The gates.**

| gate | meaning of a FAIL |
|---|---|
| SRa the two alphas agree | the intervals are DISJOINT. **Do not publish either number.** Two routes that share no byte model, no bandwidth and no estimator disagree about the parameter the tile-corrected roofline rests on, so at least one of them is measuring something other than the extra-tile cost. Check first whether the ablation's ISA gate passed and whether the regression ran in the multi-tile regime at all. If only one estimate exists the gate SKIPs and says so. |
| S8a every expected artefact exists and is non-empty | a step reported success and produced nothing. Check its log BEFORE tearing down; re-running one step is minutes and re-renting is an hour. An artefact is expected only when its producing step passed, so a skipped step never shows here. |
| S8b the calibration is byte-identical to step 1's snapshot | **STOP.** The ruler moved under the results and `publish_results.sh` would copy a calibration these rows were never measured against. Restore `$SESSION/calibration/` first. |
| S8c nothing is silently gitignored | **STOP.** `git add` drops an ignored file without a word. Rename the artefact or fix the rule. |
| S8d publish_results exit 0 | the commit may still exist locally: a public clone over HTTPS can pull but not push. `gh auth login`, or copy the session off with `runpodctl send $SESSION`. |
| S8e session artefacts tracked | the census and the logs exist on the pod and nowhere else, and they are what makes C1 and C3 checkable without a GPU. |
| S8f local HEAD equals the remote | the commit is local only. Push it or send the directory. |
| S8g MANIFEST.sha256 written | without it a truncated copy off the pod is indistinguishable from a complete one. |

**Then, and only then, stop the pod.** Everything under `$SESSION` is on the
Network Volume and survives termination. Everything under `/` does not.

---

## Reading the output

**Exit codes.** 0 every gate passed or was deliberately skipped. 1 a soft gate
failed and the session continued with a named consequence. 2 a fatal gate failed
and the session stopped. 3 the script was used wrongly.

**The ledger.** `$SESSION/LEDGER.tsv` is tab-separated: step, name, status,
observed, gate, consequence. It is what `--from` reads and what a post-mortem
should start from.

```bash
column -t -s"$(printf '\t')" "$SESSION/LEDGER.tsv"
awk -F'\t' '$3=="FAIL"' "$SESSION/LEDGER.tsv"
```

**The contract a step script honours.** It prints at least one line carrying the
word PASS and no line carrying FAIL, in one of exactly TWO shapes, matched
exactly rather than loosely:

```
  [PASS] regime: every cell is memory bound          <- group_m, tuned_vs_fallback, alias_ablation
GATE 7  PASS  the crossing moved with BLOCK_M        <- block_m_crossing_sweep
```

**This was wrong until 2026-09-01 and the correction is worth knowing about.** The
first version of the matcher looked for PASS as the first non-blank token or
straight after a colon, and it was validated against synthetic log lines rather
than real ones. Against the actual output of the three step scripts it read **0
PASS and 0 FAIL from all three**, so those gates FAILED on a perfect run and a
genuine refutation was equally invisible. The entire scientific payload of the
session had no verdict channel. The rule is deliberately not a bare whitespace
boundary, because `block_m` prints the prose line "a FAIL here is the interesting
answer" and other scripts print "2 PASS, 1 FAIL" summaries; both would
false-positive under a looser rule, and a false PASS is the one direction this
must never fail in.

So prose in a step script must not begin a line with `[PASS]` or `[FAIL]`, or with
`GATE <n> PASS`. Verified against the real output of all four scripts the session
counts, including `scripts/alias_ablation.py`, which prints 13 `[PASS]` lines on
`--synthetic refit` and 12 PASS plus 1 `[FAIL]` on `--synthetic retracted`.

`scripts/nsys_dram_probe.py` deliberately does NOT use this convention and is not
read with it: it is gated on its exit code and on the contents of `probe.json`,
because "nsys exited 0" and "nsys sampled something" are different facts and only
the report distinguishes them.

`scripts/block_m_crossing_sweep.py` offers `--fail-on-gate`, which turns its
scientific verdict into an exit code, and **step 2 passes it**. Without it the
sweep exits 0 with gates failed, so the exit-code gate would pass on a refutation
and the session's central result would have no way to report one. The consequence
line on that gate says in as many words that a FAIL there is a result and not a
crashed run.

---

## Failure playbook

| symptom | almost certainly | do this |
|---|---|---|
| PTX dump is empty | a cache hit; Triton does not recompile what it has already built | check `TRITON_CACHE_DIR` is per-run, not `$WORKSPACE/triton-cache`. P5 tests this before you spend anything. |
| "Using ..." config line missing | vLLM logs it once per `(E,N,dtype,device)` via `info_once` | make sure the cell is the first `fused_experts` call in the process, and that info-level logging is on |
| every efficiency column is empty | no calibration resolved for this device | `python scripts/calibrate_hardware.py`; the file resolves by device NAME |
| sweep says the calibration is foreign | `measured_<device>.yaml` was overwritten between sweep and publish | restore `$SESSION/calibration/`, then publish |
| `ncu` says ERR_NVGPUCTRPERM | the host module flag is not set, which a container tenant cannot set for itself. This row said "expected" until 2026-09-10, when the probe read OPEN on a RunPod H200; **that reading is retracted (2026-09-15)** because the probe profiled `/bin/true` and so never attempted a counter read, and a rented H200 then refused the read with `ERR_NVGPUCTRPERM` on a pod holding neither `CAP_SYS_ADMIN` nor `CAP_PERFMON`. The committed record is two refusals on two rentals, 2026-08-25 (`profiles/q2_kernel_names.txt`, `ncu` over the harness's own CLI) and that one; the 2026-09-09 and 2026-09-10 pods were never asked, session 4's probe found no ncu on PATH, and session 5 attempted none. Two pods are not the platform, so read the probe rather than expect either answer; and the host flag is only half of it: ask the provider for `--cap-add=PERFMON` first, `--cap-add=SYS_ADMIN` second | the alpha(G) chain runs this probe as its `counter-probe` step, after the preconditions and on every pass, and looks for ncu off PATH too (its `COUNTERS` file). By hand, run `python scripts/dram_counter_route.py --probe` FIRST: it launches one real CUDA kernel under ncu, distinguishes the four failures that look identical from a log, and costs about fifteen seconds (the probe child's torch import). If it reads BLOCKED, use `nsys`, the measured ceilings and the L2 flush axis (`docs/RUNPOD.md`); if it reads OPEN, the two `counter-*` arms are bookable, `counter-n32-m64` and `counter-n128-m64`, and `counter_contrast` reads them. The arm named `counter` has not existed since 2026-09-10; the identical sentence 200 lines above was updated and this one was not. |
| override_config appears to do nothing | the hook moved between vLLM versions | P4 catches this. `try_get_optimal_moe_config` reads it via `get_config()`, so it exists under some name |
| a step crashed on `--out` / `--out-dir` | a step script renamed a flag | P11b catches this. Fix the invocation in `scripts/pod_session.sh` |
| the whole script is a bash syntax error | an apostrophe inside a heredoc that sits inside `$( )` | bash 3.2 tracks quotes through it, and reports the error hundreds of lines away. No apostrophes in those blocks. |
| P-nsys says INFERRED | almost always: this pod has no DRAM sampler, which is the expected case | nothing. The session is unaffected and every traffic figure says INFERRED. Read the CONTROL rung in `$SESSION/logs/nsys_probe.log`: it separates "no sampler" from "no nsys". |
| `PnsysCal` FAILS | the sampler answered and got a KNOWN answer wrong | do not quote anything measured through it. The session continues with traffic INFERRED. This is the failure the calibration exists to catch and it is worth more than the measurement would have been. |
| step 2b exits 3 | not enough free memory, or no triton | the deepseek-v3 rung needs about 16 GiB free. Run it after the card is idle, or with `--models mixtral-8x7b,qwen2-57b-a14b,deepseek-v2-lite`. |
| step 2b says NOT TESTABLE (exit 4) | the bracket is too wide to separate 0.10 from 0.558 | read the printed `r`, the aliased ladder's per-tile cost. It sets the bracket width, and a wide bracket means L2 service is a large share of the cost. Do NOT pick a candidate from a wide interval. |
| the two alphas DISAGREE | one route is measuring something other than the extra-tile cost | check the ablation ISA gate first, then whether step 3 ran in the multi-tile regime. Do not publish either number until it is resolved. This is a result, not a bug. |
| `--from 2b` is rejected | `--from` takes a number | `--from 2` includes 2b; `--only 2b` runs it alone. |

---

## What this session does not settle

Worth keeping in view, because a good result here is easy to over-read.

- **The nsys route is UNVERIFIED, and its supporting arithmetic is
  doc-sourced.** No part of the probe has run against a GPU, and the 200 kHz
  ceiling every sampling figure rests on is NVIDIA's documented number rather
  than one this project has observed. Cite `observed_sample_hz` from the probe's
  own output rather than the constant, and check `sample_rate_honoured` before
  quoting any resolution verdict. See the provenance table in the P-nsys
  section.
- **DRAM traffic is modelled unless P-nsys says otherwise, and even then only in
  aggregate.** Pre-flight now tests the `--gpu-metrics-device` route rather than
  leaving it untested, and the session labels every traffic figure MEASURED or
  INFERRED accordingly. But sampling is not a per-launch counter substitute and
  must not be described as one: a single MoE kernel launch is not measurable at
  any rate nsys offers, and what a working sampler buys is aggregate traffic over
  a long contiguous run of back-to-back launches merged into one window. That is
  enough to validate or refute the compulsory byte model at roughly the 1% level
  and enough to choose between 0.558 and 0.10. It is not enough to attribute
  bytes to a launch, and it is device-wide, so a neighbour process on the pod is
  bounded by the idle baseline rather than excluded.
- **One GPU holding every expert.** DeepSeek runs decode on DP144+EP144 precisely
  to scale the aggregate batch past this ridge, and at that scale all-to-all
  communication dominates rather than the GEMM. Half the corrections in this
  project came from stating a single-node result as universal.
- **The regime is pure decode at modest concurrency.** At a few thousand tokens
  per forward, three of four models are already compute-bound. Crossing over in
  pure decode needs 316 concurrent sequences for mixtral and 3,010 for
  deepseek-v3.
- **The sweep is unsharded and unquantized.** Every cell is TP=1. Real serving
  shards, which changes `N` and therefore the config lookup, the tile and the
  block count.
- **The offload regime is untouched.** No host-to-device transfer has ever been
  measured here and the byte model has no offload path.
