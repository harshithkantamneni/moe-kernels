#!/usr/bin/env bash
# Every open experiment, as ONE unattended pod run, in the order their results
# are READ.
#
#   bash scripts/h200_gaps_session.sh --dry-run     # laptop, free: every arm's
#                                                   # own plan, its cost, the MDE
#   bash scripts/h200_gaps_session.sh               # the pod run. --dry-run
#                                                   # prints the total and the
#                                                   # minute each arm starts at
#   bash scripts/h200_gaps_session.sh --list        # the arms and what each closes
#   bash scripts/h200_gaps_session.sh --only calibrate,roofline-n256-g16,noise_floor
#                                                   # a subset. calibrate belongs
#                                                   # in EVERY subset: the ruler
#                                                   # gate below is not scoped to
#                                                   # --only, because nothing that
#                                                   # measures is either.
#   bash scripts/h200_gaps_session.sh --resume-latest
#                                                   # RESUME the newest session
#                                                   # for this card that holds a
#                                                   # measuring ledger (ARMS.tsv).
#                                                   # Finished arms in it are
#                                                   # skipped; that skip is the
#                                                   # latch, and it lives in the
#                                                   # session directory and
#                                                   # nowhere else.
#   SESSION=/workspace/session/gaps-<card>-<stamp> bash scripts/h200_gaps_session.sh
#                                                   # the same, naming the
#                                                   # directory outright. SESSION=
#                                                   # is the only way to resume
#                                                   # into a directory that is
#                                                   # not the newest.
#   bash scripts/h200_gaps_session.sh --new         # open a fresh session even
#                                                   # though one exists. Without
#                                                   # it a MEASURING run that
#                                                   # finds a session for this
#                                                   # card REFUSES rather than
#                                                   # opening an empty ledger
#                                                   # beside a full one, because
#                                                   # an empty ledger re-runs
#                                                   # every DONE arm and every
#                                                   # CLAIM_FAIL and INVALID row
#                                                   # the latch exists to keep.
#                                                   # A --dry-run is free and
#                                                   # skips nothing, so it opens
#                                                   # a fresh session without
#                                                   # being asked.
#
# WHAT THIS FILE IS. A schedule and a ledger. It measures nothing itself: every
# number comes from the script an arm runs, and every verdict comes from that
# script's own gates. What this file owns is the ORDER, the exit-code contract,
# and the closing summary -- and each of those three has already been wrong in a
# way that cost a rented pod.
#
# WHAT CHANGED ON 2026-09-02, second rewrite, after the standards audit. Seven
# defects, each of which made a broken run look like a clean one:
#
#   * THE EXIT-CODE CONTRACT WAS INVERTED AND PER-ARM. This file read 2 as
#     REFUSED and kept a hand-maintained list of "done" codes per arm, while
#     memory_branch_anchor.py documented the opposite (2 = a VALIDITY gate
#     failed AFTER an eight-minute measurement, 3 = nothing measured) and three
#     other scripts refused with 3. A measured-and-invalid anchor run was
#     therefore logged REFUSED and printed "REFUSED BEFORE MEASURING", and
#     genuine refusals were queued as retries. Both lists are gone. There is one
#     table, in moe/bench/exit_codes.py, and `ledger_state` below is its shell
#     mirror; tests/test_h200_gaps_session.py asserts the two agree code for
#     code, so they cannot drift.
#   * THE SUMMARY GREPPED PROSE. `floor|sigma` matched a REFUSED noise-floor log
#     eighteen times and printed `floor: 0.0905` -- an IMPORTED prior, not a
#     measurement -- and `C1 ... [PASS]` -- a pre-registered expectation, not a
#     result -- under the heading "THE GATES". The summary now greps `^RESULT: `
#     and nothing else, the one line moe/bench/exit_codes.result_line renders and
#     parse_result_lines reads back. An arm that printed no RESULT line was not
#     scored, and the summary says exactly that instead of finding words.
#   * A TRACEBACKING --dry-run LOOKED LIKE A CLEAN PLAN. Every dry-run exit code
#     was mapped to PLANNED. A plan now records its rc and gets one of three
#     words, and the session exits non-zero if any plan is BROKEN.
#   * THE CARD WAS MISSING FROM THE RESULTS ROOT. The comment said the card was
#     "IN BOTH NAMES" while the export was a bare /workspace/results, which
#     outlives the pod: a second card resumed the first card's directories and
#     published the first card's timings under the second card's heading. That
#     is a committed collision, not a hypothetical (provenance.py, collision 2).
#   * THE HEADLINE ARM COULD NOT CONFIRM ITS OWN CLAIM. It ran BLOCK_N=64,
#     GROUP_SIZE_M=1 -- the study's swept configuration -- under a heading about
#     what production runs. vLLM 0.27.1's tuned entry for this shape
#     (moe/bench/hardware/vllm_configs/E=8,N=14336,device_name=NVIDIA_H200.json)
#     is BLOCK_N=256 with GROUP_SIZE_M=16, and 32 at 2048 tokens. Three roofline
#     arms now run: the swept-configuration control, which can REFUTE and not
#     confirm, and the two production configurations, which were meant to
#     confirm and CANNOT on sm_90: bm128_roofline refuses them at every warp
#     and stage count (a 256x256 fp32 accumulator is 65536 of 65536 registers
#     per block), so no BLOCK_SIZE_N confirms the headline on this card.
#   * THE PIN WAS PROBED AT A CONFIGURATION NO ARM RUNS. The probe pinned
#     BLOCK_N=128 while every arm pins 64 (or, now, 256). A pin that reaches the
#     kernel at one BLOCK_N is evidence about that BLOCK_N.
#   * THE ONE TABLE WAS ADOPTED HERE AND NOWHERE ELSE, SILENTLY. Adopting
#     moe/bench/exit_codes fixed this file's reading and fixed none of the
#     twenty scripts it reads. That has since been closed for the measuring
#     arms: every script this driver runs to time a cell imports the module,
#     which is why `adopts_exit_codes` is a live check per row rather than a
#     list maintained here. What remains unadopted is the analysis and probe
#     tail, and one of them is an arm: dram_counter_route.py returned 3 for
#     every verdict that was not
#     OPEN -- so a BLOCKED counter route, which is that arm's registered ANSWER
#     on a rented pod, landed in the ledger as INVALID and was described to the
#     operator as measured-and-unquotable. The dry-run said this, for free,
#     where it costs nothing; the pod run said nothing, where it costs an arm.
#     (That script adopted the table later the same day, and since 2026-09-03
#     it scores one gate per verdict and prints one RESULT line each, so the
#     ledger word is EARNED: OPEN lands DONE, BLOCKED lands CLAIM_FAIL and is
#     LATCHED as the answer, and REFUSE -- no ncu on PATH, or nsys without its
#     importer -- exits 2 with no RESULT line and is re-attempted on every
#     resume. `arm_closes counter_plan` says the same; an earlier version of
#     this paragraph said it exited DONE on BLOCKED with no RESULT line, which
#     was true for one day and contradicted `arm_closes` for five.)
#     The states are NOT patched per arm -- that is the list R1 deleted. Instead
#     `adopts_exit_codes` ASKS each arm's file whether it imports the module,
#     and `contract_caveat` / `contract_disclosure` print, next to every REFUSED
#     or INVALID row that came from a file which does not, what the state may
#     actually mean and in which direction. When a sibling slice lands the fix,
#     the caveat stops printing on its own; nothing here has to be remembered.
#
# WHAT CHANGED ON 2026-09-02, third pass, after the whole-repo verification.
# Five more, and the first two were introduced by the rewrite above:
#
#   * THE SESSION NEVER PUBLISHED THE CALIBRATION IT MEASURED. Arm 0 ran
#     calibrate_hardware.py bare. That script's default output had moved to an
#     untracked session path and the copy into the tracked tree had become a
#     separate --publish decision, so arm 0 spent three minutes measuring THIS
#     card's ridge into a directory nothing reads while arms 1-10 resolved
#     theirs from whatever measured_<card>.yaml the LAST rental left in the
#     checkout -- and labelled it "measured on this machine". That is the
#     audit's own "a constant from another machine presented as a
#     measurement", recreated by
#     the fix for a different one, and the closing `git diff --stat
#     moe/bench/hardware/` would have shown zero changes and read as agreement.
#     The flag is passed now, and the gate that follows arm 0 REFUSES the rest
#     of the session unless BOTH halves answer: the tracked yaml for this card
#     carries a provenance.utc at or after this session's own start, AND the
#     calibrate row in this session's ledger reads DONE. The second half is not
#     redundant -- calibrate_hardware.py publishes the yaml BEFORE it scores a
#     gate, so a run that lands INVALID on clock_established leaves a
#     fresh-stamped ruler nothing verified behind it, and the first version of
#     this gate read that as "measured in THIS session". The committed H200 yaml
#     carries no provenance block at all, so it reads UNDATED and does not pass.
#   * THE TWO SPAN ARMS DERIVED ONE RUN ID. --densify became the default while
#     the driver still booked `span_dense` with it and `span` bare, so both
#     densified, both hashed to the same plan, and the second restored the
#     first's rows, measured nothing and landed DONE with 30 minutes booked.
#     The sparse arm names --no-densify now, which is in the key.
#   * THE DISCLOSURE COVERED TWO STATES AND THE NON-ADOPTERS SPEND NEITHER.
#     contract_caveat and contract_disclosure fired only on REFUSED and INVALID
#     while check_mma_path.sh spent 1 on a failed gate and moe/bench/cli.py
#     spent 4 on one, so a ledger holding only `mma_switch CLAIM_FAIL 1`
#     printed the ALL-CLEAR over the one row whose word was wrong. Which files
#     have adopted is still ASKED of the files, so a sibling slice landing a
#     fix turns the disclosure off by itself.
#   * AN EXIT 1 THIS FILE CANNOT READ WAS LATCHED AS A RESULT. Python spends 1
#     on any exception that escapes main, and at this pass three of the
#     thirteen Python files this driver runs install the ERROR(4) top-level
#     handler that would say so (calibrate_hardware, bm128_roofline,
#     memory_branch_anchor), so a crash in any of the other ten was filed as
#     CLAIM_FAIL, "a result, never re-run", and the arm was skipped forever. An
#     exit 1 from a file that has not adopted the table is recorded UNKNOWN and
#     not latched.
#   * DRY MODE COULD NO LONGER TELL A PLAN FROM A REFUSAL. Once every script
#     adopted the table its --dry-run began exiting REFUSED, correctly, and
#     eleven of the sixteen planned arms collapsed onto PLAN_REFUSED. `dry_state` reads
#     the log now: a plan that printed and then refused is PLANNED, a refusal
#     with nothing printed before it is PLAN_REFUSED.
#
# WHAT CHANGED ON 2026-09-03, fourth pass, after the pod-readiness check ran
# every arm exactly as the lines below invoke it. Its verdict was NOT WORTH
# RENTING AS SCHEDULED, and six defects carried it. Every one of them was a
# number or a flag in THIS file disagreeing with what the arm's own script does:
#
#   * THE NOISE FLOOR WAS BOOKED AT A TENTH OF ITS OWN PLAN, AND PUBLISHED
#     NOTHING. `replicate_noise_floor.py --dry-run` prints "4 arm(s) x 6
#     replicates ... TOTAL: ~240 min", and this file booked 25 and ran the
#     script bare. Arm 8 of 20, arms are sequential, and nothing here has a
#     deadline: a session sized off the old table reached minute 12, entered the
#     floor, and was killed inside it when the pod was released -- which fails
#     that script's own V2 ("the planned replicates ran"), so even the partial
#     floor was unquotable, and arms 9-20 never started. Bare also meant no
#     --publish, and NOISE_FLOOR.json is written under that flag alone, so the
#     most expensive arm in the session left the sigma every MDE line prints
#     still reading ASSUMED. The arm now names --replicates 3, its four arms and
#     --publish, and its booking is what its own --dry-run prints for those
#     flags. --publish is NOT on the dry-run line: with it, a --dry-run writes
#     results/published/NOISE_FLOOR.json, a TRACKED file, which is the anchor
#     rescore's defect in a second place.
#   * THREE ARMS COULD NOT REACH THE ONE STATE THEY WERE SCHEDULED FOR.
#     occupancy and both bn arms ran without --fail-on-gate, and
#     occupancy_vs_swizzle.exit_for and bn_decomposition.exit_for BOTH downgrade
#     a CLAIM_FAIL to exit 0 DONE when it is absent. Live, off GPU:
#     `occupancy_vs_swizzle.py --audit` prints "RESULT: CLAIM
#     A1_reuse_distance_on_the_corpus FAIL" and exits 0, while
#     exit_codes.classify_text over that same log returns 1. So a failing claim
#     was filed DONE, contract_disclosure printed the ALL-CLEAR over it because
#     the import is real, and the CLAIM_FAIL branch this header spends four
#     paragraphs on was dead for the arms whose registered outcome is a failing
#     claim. Every arm whose script defines a gate flag is now given it, and a
#     test walks the invocations rather than a list kept here.
#   * ONE ARM WAS SCHEDULED AT A PINNING ITS OWN DESIGN GATE CALLS INVALID.
#     bn_g1 ran GROUP_SIZE_M=1, where `bn_decomposition.py --self-test
#     --capability 9.0 --group-m 1 --reps 17 --plant-noise 0.008` exits 3
#     INVALID: S4 sees sd(alpha_a) 0.1759 against a gate of 0.025, and S5 sees
#     the planted MISSING world pass C2 at chi2 1.78 against a ceiling of 4.0,
#     so neither alpha_a nor C2 can be resolved there however the data fall. The
#     arm is DROPPED, and the reason is that no re-pinning exists: the same
#     self-test fails at every GROUP_SIZE_M this instrument was checked at
#     except 16 (1, 2, 4, 8, 32 and 64 all exit 3; sd(alpha_a) 0.1759, 0.1759,
#     0.1759, 0.4864, 0.1759, 0.0635), and 16 is the arm already scheduled. Its
#     36 priced minutes go back to the session. What it was reframed to carry --
#     alpha_b, C3 and C5 at the swizzle every published arm swept -- had no
#     registered detection limit at that pinning either: S1 checks alpha_b's
#     LEVEL on one planted draw and nothing bounds its SPREAD there.
#   * THE COST TABLE WAS WRONG BY 2.1x OVERALL AND 10x ON ONE ARM. Every
#     arm_minutes entry is now what THAT arm's own plan prints when invoked the
#     way the lines below invoke it, `arm_basis` names the command and the
#     figure so an operator can re-derive any row in seconds, and `arm_unpriced`
#     carries, in the plan's own words, what the figure leaves out. The two
#     BLOCK_N=256 rooflines and the sparse span arm are booked ZERO because
#     their plans refuse before any GPU time, which is a result and not a cost.
#     And TOTAL is printed in BOTH modes: it used to sit inside the `else` of
#     `if (( DRY ))`, so the one number that decides whether to rent was printed
#     only after the decision. Each arm now also prints the cumulative minute it
#     starts at, so a rental length can be read off the table.
#   * THE DRY RUN PREVIEWED A DIFFERENT RUN THAN THE POD EXECUTES, in four
#     arms. calibrate was skipped with a reason that is false -- "has no
#     --dry-run"; it has one, prints a nine-line plan and exits REFUSED -- and
#     it is the arm whose gate can end the session at minute 3, so its plan is
#     the one an operator most needs before renting. anchor_measure dry-ran
#     without --measure and so previewed rescore mode and 26 committed reports
#     rather than "cells 128 ... estimated wall time 4.8 min". bm128_depth
#     dry-ran at the default --r-max and previewed 126 s and a -r1024- run id
#     while the pod runs --r-max 2048, 252 s and -r2048-. dtype dry-ran without
#     --card, refused with NoCardToLabel and landed PLAN_REFUSED, which reads as
#     a broken arm rather than as this laptop having no GPU; it is given a
#     NAMED HYPOTHETICAL card off GPU, labelled as one, the way pin_probe
#     already distinguishes the two.
#   * span's ADVERTISED OFF-GPU CHECK SCORED NOTHING, and its pod run was
#     truncated mid-sweep. `span_extent_separation.py --self-test kernel
#     --densify` exits 2 with ZERO RESULT lines in all three worlds and closes
#     with "Pass --fail-on-world to score the S gates"; with the flag it is 15
#     RESULT lines and exit 0 in each. The advertised command carries it now.
#     Separately, span_dense ran with --max-minutes 35 against a plan that
#     prices 1814 s of KERNEL alone and says the wall clock is not that number.
#     --max-minutes does not refuse: it breaks out of the cell loop, records
#     "stopped after N minutes with K of M cells done" as PROSE, and the gates
#     then score the truncated grid to the same exit code a complete one gets.
#     The flag is gone from both span arms and the honest time is booked. THE
#     REAL FIX IS NOT HERE: a truncation that is scored as if complete has to
#     become a REFUSED where the loop breaks and where the gates are built, in
#     span_extent_separation.py, because only that file knows which cells are
#     missing and whether the reduction can stand without them. The only lever
#     this driver has is whether to arm the truncation at all, and arming it
#     converts a complete-grid arm into a partial one wearing a complete one's
#     exit code.
#   * THE TOTAL ADDED TWO DIFFERENT CLOCKS, and the "starts at" column -- the
#     one thing an operator sizes a rental with -- was computed from that sum.
#     Fixing every figure to be the arm's own left them in mixed units:
#     noise_floor's 120 and anchor_measure's 5 are WALL (replicate_noise_floor.py
#     scales its 3066 s model by the 2.35x it measured; memory_branch_anchor.py
#     prints "estimated wall time" and charges a compile per setting), while
#     seven arms are booked at what their plans call "the model's own timings,
#     excluding compiles and allocation": roofline-n64-g1 58 s, bm128_depth
#     252 s, bn_g16 2754 s, occupancy 1342 s, cap_test 242 s, dtype 454 s and
#     span_dense 1814 s. The paragraph under the table then said BOOK ABOVE THAT
#     AND NEVER AT IT and gave no number to book above, leaving the mixed sum as
#     the only figure on the page. Every row names its clock now, and the total
#     is printed twice: what the plans price, and the same table with its KERNEL
#     part multiplied by the ONE wall-over-model ratio this repo has measured
#     (127 s logged against 54 s modelled for mixtral_g1, the factor
#     replicate_noise_floor.py already applies to its own booking). That second
#     number is an ILLUSTRATION OF THE SIZE OF THE GAP from one small arm and not
#     an estimate of any arm; no per-arm figure is multiplied by anything, which
#     is the same refusal as before, now with the bound stated instead of implied.
#   * THE dtype OFF-GPU CHECK EXAMINED NOTHING, which is the span defect above
#     repeated inside the commit that named it. The fix for a script that needs
#     a card to name went into the dtype DRY-RUN branch and not into
#     `arm_offgpu_gates`, so the command an operator runs before renting --
#     `dtype_tile_confound.py --self-test 2.033 --self-test-alpha 0.2` -- exits 2
#     with NoCardToLabel and ZERO RESULT lines in all three planted worlds, and
#     so does the --dry-run half of the same line. With --card it is 10 RESULT
#     lines per world and C3 separates them: 2.033 PASS 1.023, 2.400 FAIL 1.208,
#     1.000 FAIL 0.503. A guard walks every advertised off-GPU command now and
#     runs it, rather than reading it.
#
# WHAT CHANGED ON 2026-09-03, fifth pass, after the final verdict read this
# schedule against the study's own inferential chain rather than against its own
# arithmetic. ONE defect, and it is an ABSENCE rather than a wrong number:
#
#   * THE ARM THAT TESTS THE STUDY'S FIRST INFERENTIAL LINK WAS IN NO SESSION.
#     `grep -c alias_ablation scripts/h200_gaps_session.sh` returned 0, and the
#     same grep over all 28 branches in this repository returned 0 on every one.
#     Every alpha here is a slope per extra M-tile RELABELLED as a fraction of a
#     fresh DRAM weight read, and every cap, every roof fraction and the
#     sentence "a decode-configured MoE kernel can never reach its compute
#     roof" is that
#     relabelling carried forward. Nothing in the schedule tested the
#     relabelling. scripts/alias_ablation.py is the only instrument that can: it
#     measures the same physical quantity with no compulsory-byte column, no
#     calibrated bandwidth, no ridge and no fitted intercept, by running one
#     access pattern twice and pointing one arm's weight loads at an L2-resident
#     column block. The arm is scheduled now, at 3, ahead of the 120-minute
#     noise floor, for the reason in the order list below.
#   * AND IT IS 13 WALL MINUTES, NOT THE HOUR THE VERDICT RESERVED FOR IT. Asked
#     of the script rather than assumed: `alias_ablation.py --card "NVIDIA H200"
#     ... --run`, on a box with no GPU, prints "WALL 13.7 min ... BOOK THIS ONE"
#     and then refuses at the probe having measured nothing. It is also the one
#     arm in this session whose DRY preview UNDER-BOOKED its own pod run
#     until 2026-09-03: `report_cost` charged the probe's ten specialisations
#     only when --run was named, so the bare plan said 11.6 where the pod spent
#     the full figure. alias_ablation now charges the probe on the plan page as well,
#     because an operator books a pod before they have one and the plan is the
#     only page they can read; the bare plan and the pod run print the SAME
#     figure. The dry branch stays bare (a --dry-run carrying --run would
#     MEASURE on a pod, and a plan must be free in every sense), and `arm_basis`
#     names the command and says the two agree, so the row can be re-derived
#     in seconds. The gap is closed, not disclosed.
#
# WHAT CHANGED ON 2026-09-03, sixth pass, after a reviewer read the alias arm's
# two OPERATOR surfaces against the sibling script rather than against the body
# comment above the arm. ONE defect, and it is this repo's recurring shape: a
# state disclosed where the arm is PLANNED and at neither place it is REPORTED.
#
#   * THE LIKELIEST READING OF THIS ARM'S EXIT 1 WAS IN NEITHER SURFACE.
#     `arm_closes alias_ablation` and the READ-FIRST block both enumerated three
#     states -- P1 PASS, P1 FAIL, headroom/attribution INVALID -- and glossed
#     exit 1 as the FAIL ("the interval says which of 0.10 or 0.33 it landed on
#     instead"). The arm was booked `--dot-fallback allow` that day (it is
#     `refuse` since 2026-09-09, below), and
#     `alias_ablation.choose_pinning` calls the fall to dot mode the LIKELY case
#     rather than the corner: the 0.61-of-roof ceiling this arm exists to escape
#     has the signature of the cross-lane `tl.sum` tree that `dot` removes. A
#     dot ladder measures a LOWER BOUND, leaves P1 UNKNOWN, `classify` maps an
#     UNKNOWN CLAIM to CLAIM_FAIL, and `arm` LATCHES a CLAIM_FAIL, so the arm
#     spends its thirteen minutes, answers nothing, and wears the word a
#     refutation wears. The gloss the two surfaces DID carry is exactly the
#     write-up the sibling script exists to prevent: "alpha is not 0.558" when
#     what happened is that alpha was not asked. Both surfaces now name the
#     fourth state, send the operator to the P1 RESULT line's verdict WORD
#     rather than to the exit code, and forbid the 0.10-or-0.33 sentence from an
#     UNKNOWN. Nothing about the booking, the order or the arithmetic moved:
#     this is a reporting defect and the fix is in what the operator reads.
#
# WHAT CHANGED ON 2026-09-03, seventh pass, after a reviewer ran this file's own
# LIFTABLE `arm()` over planted commands and read what the ledger latched. Four
# defects, and the first is the one every other pass had been describing as
# fixed while nothing called the function that fixes it:
#
#   * THE SECOND OPINION WAS NEVER TAKEN. exit_codes.classify_text exists so the
#     driver can recompute a script's verdict from the RESULT lines it printed
#     and compare it with the code the process returned, and this file said so
#     in four places, every one of them a comment or an echo. `arm` decided the
#     state from the integer alone and `summarize_arm` grepped RESULT lines for
#     display only. Proven with the driver's own functions: a log reading
#     `RESULT: CLAIM C1 FAIL` under exit 0 landed DONE and was latched; a log
#     with no RESULT line at all under exit 0 landed DONE and was latched; a
#     log of all-PASS lines under exit 1 landed CLAIM_FAIL and was latched.
#     `arm` now runs `log_verdict` over the captured log and `second_opinion`
#     over the pair. When the log HAS RESULT lines and they imply a different
#     code from the one returned, the row is a DEFECT: both codes go in the
#     ledger note, the state is UNKNOWN, nothing is latched, the closing
#     summary prints it under its own heading and the session exits INVALID
#     over it, because what is on that page is not a verdict. When the log has
#     NO RESULT lines and the code is 0, DONE is UNEARNED: a check that
#     examined nothing reports no failures, and the row is UNKNOWN. The same
#     for 3: an INVALID that names no failed gate is not latched either. The
#     one disagreement that is not a defect is a code OUTSIDE the table over a
#     scored page: the arm printed its gates and then crashed, the process code
#     wins, the row is RETRY and the note says what the page implied.
#   * AN INTERPRETER EXIT 1 WITH NO RESULT LINE WAS LATCHED AS A REFUTED CLAIM.
#     The scripts' ERROR(4) guards wrap `_main()` and cannot catch a failure at
#     import time, which is the pod's real 2026-09-01 shape (a torch whose ABI
#     had drifted). With a broken `torch.py` planted on PYTHONPATH and the real
#     pod command lines run through the real `arm()`: alias_ablation,
#     pin_probe, calibrate and dtype all exited 1 with zero RESULT lines and
#     were filed CLAIM_FAIL, latched, and skipped on every resume; for
#     calibrate that turned the calibration gate's `ARM CLAIM_FAIL` into a
#     REFUSED on every resume until someone deleted the row by hand. The
#     UNKNOWN demotion of the third pass covered only a file that has not
#     adopted the table, and every measuring arm has. scripts/check_mma_path.sh
#     already reads this exact signal for its own child and maps it to ERROR;
#     `second_opinion` applies it here: exit 1 with zero RESULT lines from a
#     file that speaks the table is a CRASH, RETRY, never CLAIM_FAIL, with the
#     tail of the log printed.
#   * THE LATCH LIVED IN ONE DIRECTORY AND NOTHING SAID HOW TO GET BACK INTO
#     IT. SESSION= was read and never documented; a plain re-invocation opened
#     a fresh stamped directory with an empty ledger, so every DONE arm was
#     re-attempted and every CLAIM_FAIL and INVALID row, the rows the latch
#     exists to protect, was re-run. Two consecutive --dry-runs produced two
#     session directories. SESSION= is in the usage block now, --resume-latest
#     picks the newest session for this card that holds a measuring ledger, and
#     a MEASURING run that finds such a session REFUSES to open a fresh one
#     unless --new is given. The decision is `session_choice`, one function,
#     so every branch of it is plantable off GPU.
#   * contract_caveat DESCRIBED THREE FILES AS THEY WERE BEFORE THEY ADOPTED.
#     It said check_mma_path.sh spends 1 on refusals (2, since it adopted),
#     that its --dry-run exits 0 (2), that cli.py returns 4 on a pin miss
#     (INVALID through exit_codes.classify) and that dram_counter_route.py
#     returns 3 on BLOCKED (DONE). Harmless only because all four adopt and the
#     caveat never prints for them; each is now told as the history it is.
#
# WHAT CHANGED ON 2026-09-09, after the first H200 session ran the 3-hour set in
# 33 minutes and a seven-agent analysis re-verified every arm by execution. The
# ledger's own count is six DONE, two REFUSED, two CLAIM_FAIL as designed, and
# SIX INVALID. Not one of the six was a fact about the card: each was an apparatus
# defect, and four of them are booked here.
#
#   * THE CLOCK RULE WAS A RULE AGAINST A TILE. Under load an H200's SM clock is
#     set PER TILE by that kernel's own power draw under the 700 W cap:
#     BLOCK_M=128 holds a median 1395 MHz over 196 cells, BM=256 1650 over
#     311, BM=32 splits by SWIZZLE (1474 at GROUP_SIZE_M=1 over 18, 1740 from
#     GROUP_SIZE_M=8 up over 50; the single pooled "1736 over 68" this line
#     carried until now is a median over a bimodal set and names neither),
#     memory-shaped cells 1950-1980, and the calibration GEMM
#     itself 1485 at 691 W, near the LOW end of dense work rather than the
#     middle. LEVEL-LOW therefore excluded the study's two primary tiles from
#     measurability on this card and nothing else: 15 of roofline's 39 cells and
#     110 of depth's 168 treads, every one of them a steady state rather than a
#     throttle. The rule is now DRIFT ONLY (`clock_drift_ok is False`), the
#     LEVEL side is RECORDED on every row and excludes nothing, compute-bound
#     CLAIM gates read the FIXED roof fraction and the own-clock fraction is
#     printed beside it as issue efficiency. In this file that lands in the
#     summary lines and in `scripts/pod_session.sh` S6d; the instrument and the
#     consumers are the other slices' files.
#   * cap_test's --r-max WAS LEFT TO A DEFAULT AND THE ARM WAS UNSATISFIABLE
#     FROM THE PLAN PAGE. See the arm, below: 688 rows, grid stops at 672, two
#     BLOCK_M=256 stacks against V1's three. Booked --r-max 2112, which is the
#     minimum the script's own V4 check prints: 162 cells and 242 s. 1024 was
#     booked first and is refused at plan time by that same check.
#   * alias_ablation SPENT 308 s TO REACH "not asked". The probe's best sum
#     reading was 5500 GB/s against a 6151 bar, --dot-fallback allow took the
#     dot ladder, and P1 came back UNKNOWN with the ledger latching INVALID
#     (ARMS.tsv: alias_ablation INVALID rc 3, 308 s; the log's last line is
#     "EXIT: 3 INVALID", level, placebo, form and bracket having all FAILed).
#     Booked `refuse` now, against a probe grid widened DOWNWARD in warps: a
#     miss costs the probe alone.
#   * bm128_depth REFUSED ITS OWN REFERENCE ON EVERY CLOCK RULE. The {128,256}
#     pairing puts the non-vacuity floor at 0.838 of the roof and no BM=256
#     ladder in the corpus reaches it; the arm was pre-registered to be INVALID
#     on both cards. scripts/bm128_depth.py carries a BM=32 scaling partner in
#     its BLOCK_SIZES now, unconditionally and not behind a flag, and the
#     vacuity floor is derived from the CONTROL'S MEASURED PLATEAU rather than
#     from the dense GEMM roof.
#   * dtype LEAKED A TRITON CONFIG ACROSS 41 ARMS. vLLM 0.27.1's
#     `override_config` has no try/finally (verified from the tag), a Triton
#     OutOfResources inside it left the fp8 config installed process-wide, and
#     the cross-config arm is infeasible at 22 of 28 cells by shared-memory
#     arithmetic. The arm is re-scoped to transplant BLOCK_SIZE_M and
#     GROUP_SIZE_M only; what it closes and what it costs say so here.
#   * THE PREFLIGHT CLOCK PROBE PRINTED THE OPPOSITE OF WHAT IT MEASURED, and
#     measured the wrong reader. `nvidia-smi ... | grep -q` under `pipefail`
#     returns 141 on a MATCH; and the arms read clocks through
#     `torch.cuda.clock_rate()`, not through nvidia-smi. Both fixed at the
#     probe.
#   * A SESSION OF INVALID ROWS CANNOT BE RESUMED, and this file said "resume"
#     anyway. --resume-latest re-runs no CLAIM_FAIL and no INVALID row: that is
#     the latch working. The next session after an INVALID set is `--new`, and
#     the closing summary now prints that command with the arms and the state
#     each is expected to reach.
#
# WHAT A 2-HOUR AND A 3-HOUR RENTAL ACTUALLY REACH, since adding an arm is also
# a claim about what still fits. The whole session does not fit in either, and
# it did not before this arm was added: the alias hour is 13 minutes and it runs
# at minute 13, while the noise floor alone is 120 WALL minutes and everything
# the owner calls the scientific payload sits BELOW it in the read order. So the
# session prints two named subsets under its cost table, both computed from
# `arm_minutes` and `arm_clock` so that a re-booked arm moves them by itself:
#
#   2 HOURS  calibrate, both pin probes, the three rooflines, alias_ablation and
#            bn_g16. Both payload arms with their preconditions, nothing else.
#   3 HOURS  the same plus bm128_depth, the anchor pair, mma_switch, ruler,
#            cap_test, dtype and counter_plan. Still no noise floor.
#
# THE FLOOR IS WHAT DOES NOT FIT, AND STARTING IT IS WORSE THAN SKIPPING IT. A
# rental that enters a 120-minute arm it cannot finish is killed inside it,
# which fails replicate_noise_floor's own V2 ("the planned replicates ran") and
# makes even the partial floor unquotable -- the exact failure the fourth pass
# above records. The floor wants a booking of its own, and until it has one
# every effect this study reports is scored against the ASSUMED sigma the MDE
# line below names in that word.
#
# THE THREE FINDINGS THAT SET THE ORDER, restated because two of them were
# retracted since this file last said them:
#
#   * THE CAP IDENTITY IS RETRACTED. "cap = 2*BM/(alpha_fitted*b) is EXACTLY the
#     three-term cap, so the published cap numbers stand" was a tautology: the
#     test that checked it defined alpha_fitted by the formula it was testing.
#     The estimator actually in use is LadderFit.alpha = B/(A+B), which is
#     (alpha_b + phi)/(1 + phi + delta) -- the (EXA) form in
#     moe/bench/ai_model.py -- not the (LIN) blend the identity assumed. A cap
#     built from a fitted alpha is high by (1 + phi + delta): 31% at
#     BLOCK_M=128, which is larger than the cap-to-ridge gap it was being used
#     to decide. No published cap number is quotable until it is re-derived
#     through (EXA) on ladders re-timed with moe/bench/timing.time_kernel.
#   * THE REGIME, counted on max rows -- what moe_align_block_size actually pads
#     to -- and on UNIFORM routing: BLOCK_M=128 is the only tile vLLM runs
#     multi-tile in more than one cell, 65 of 87 cells, up to 33-34 tiles per
#     expert. BLOCK_M=16 (1 of 24) and BLOCK_M=64 (5 of 112) fire in isolated
#     cells. SKEWED ROUTINGS WERE NEVER COUNTED and production routing is
#     skewed, so this is a statement about uniform routing and about nothing
#     else. The earlier "59 of 87, up to 32, and 16/32/64 never" was counted on
#     seed rows rather than padded rows.
#   * TEMPO states the tile term is inactive in decode. Contesting that sentence
#     with a measurement is what this study has to offer, and it lives at
#     BLOCK_M=128 in the configuration production ships, which is why the two
#     BLOCK_N=256 arms exist and why the BLOCK_N=64 arm is labelled a control.
#
# THE ORDER IS THE ARGUMENT, so it is stated before the code. The rule behind it:
# anything whose result changes how a later arm is READ runs before that arm.
#
#   0 calibrate      THIS pod's own ceilings. Not optional: five of the arms
#                    below REFUSE without a calibration for the attached device,
#                    and the H200's dense bf16 moved 7.1% between two sessions
#                    while its bandwidth held to 0.014%, so the ridge is not a
#                    constant you can carry over.
#   0 pin_probe-*    Does MOE_FORCE_TILE reach the kernel on this build, AT THE
#                    CONFIGURATION THE ARMS RUN -- once at BLOCK_N=64 for the
#                    control arm and every alpha arm, once at BLOCK_N=256 for
#                    the production arms. That is the 2026-09-01 S6a failure,
#                    and a probe at a third BLOCK_N would have been evidence
#                    about a configuration nothing runs. Cheap.
#
#   1 roofline-n64-g1     THE CONTROL, not the claim. BLOCK_M=128 forced at the
#                    study's swept configuration (BLOCK_N=64, GROUP_SIZE_M=1).
#                    Production never ships this shape, so a plateau here
#                    REFUTES nothing about production and CONFIRMS nothing about
#                    it either: if this reaches the roof, every richer
#                    configuration does too, and arms 2-6 are measuring a
#                    ceiling that never binds. That asymmetry is the whole
#                    reason it runs first, and its predicted outcome from the
#                    published G=1 ladders is already known (gap 0.02-0.06
#                    against a 0.10 separation threshold: NOT TILE-ATTRIBUTABLE).
#   1 roofline-n256-g16   THE CLAIM'S CONFIGURATION. What vLLM actually ships
#                    for mixtral at BLOCK_M=128 on the H200, at every token
#                    count from 512 up. It was scheduled as the one arm that
#                    could CONFIRM the ceiling; it CANNOT on sm_90. bm128_roofline
#                    refuses it AT EVERY WARP AND STAGE COUNT, free and before
#                    the pod: the BLOCK_M=256 control's 256x256 fp32 accumulator
#                    is 65536 of 65536 registers per block however the warps
#                    are split, no pin rescues it, and NO BLOCK_SIZE_N confirms
#                    the headline on this card. The refusal is the arm's
#                    finding and the reason the paper has no confirming arm;
#                    see the note above the arm.
#   1 roofline-n256-g32   The same at the swizzle vLLM ships at 2048 tokens,
#                    which is inside the multi-tile range the claim is about.
#                    Refuses for the same accumulator; zero minutes.
#   2 bm128_depth    The regime every other arm is read in. The whole 128 row
#                    currently rests on two fits across two cards, one of them
#                    on a non-monotone ladder that should have been discarded.
#   3 alias_ablation  THE FIRST INFERENTIAL LINK, and the cheapest arm in the
#                    session that can void the most. It asks whether the
#                    per-tile slope every alpha is built from IS DRAM traffic,
#                    by a route that touches no byte model. It runs BEFORE the
#                    floor and before every alpha arm: its result changes how
#                    bn_g16, the anchor, occupancy, cap_test and both span arms
#                    are READ -- whether they are decomposing a fraction of a
#                    weight read or a fraction of something else -- while the
#                    floor changes only how each of them is SCORED. The two are
#                    independent (this arm states its own MDE from its own
#                    pass-to-pass scatter and reads no NOISE_FLOOR.json), so the
#                    order between them is a budget decision, and it is taken in
#                    the direction that buys the payload: a 13-minute arm that
#                    can retire the mechanism sentence does not sit behind a
#                    120-minute one.
#   4 noise_floor    Nothing above it can be scored without it. The study has
#                    NO true replicates; its closest proxy confounds num_stages
#                    and is the sigma the MDE line below is forced to assume.
#                    THE LARGEST ARM IN THE SESSION AND DELIBERATELY SO, bounded
#                    at --replicates 3 rather than the default 6 and published
#                    with --publish, which is the whole point of spending it.
#                    Its four arms are two models x two swizzles, which is the
#                    SMALLEST set that scores its own V7 (a floor measured only
#                    where the swizzle effect is 0.3855 does not license a
#                    surface spanning models where it is 0.02) and both C3s (a
#                    swizzle contrast needs G=1 AND G=16 of the same model).
#
#   5 bn_g16         The only clean separation of alpha_a from alpha_b -- BN
#                    appears in one term of the blend and BM in two -- and the
#                    test of whether the model is COMPLETE: three terms means a
#                    straight line in BM/BN, and structure in the residual names
#                    the missing one. THE G=1 PARTNER IS GONE, not demoted: at
#                    GROUP_SIZE_M=1 that script's own design self-test exits 3
#                    INVALID, alpha_a's spread is 0.1759 against a gate of
#                    0.025, and the planted MISSING world passes C2 at chi2
#                    1.78 against a ceiling of 4.0, so neither of the arm's two
#                    readouts can be resolved there however the data fall. The
#                    same self-test fails at every other pinning it was checked
#                    at except 16, so there is nowhere to re-pin it TO.
#   6 anchor         alpha_fitted's LEVEL, which the cap divides by. In 12 of 12
#                    fits the measured n=1 tread sits above the fitted branch;
#                    three defensible anchors give 0.45/0.65/0.71 for one cell.
#                    A wrong level is a wrong cap, and at 128 the cap is
#                    knife-edge.
#   7 occupancy      Does the standard predictor transfer. Reuse distance says
#                    G=64 should cut the weight re-read to 0.016; measured 0.67.
#                    If alpha tracks RESIDENCY rather than program order, the
#                    swizzle is a dead lever, the cross-card null is explained
#                    (both caches saturated), and TileSight's method does not
#                    apply in this regime -- a correction, not a re-derivation.
#                    P2 is EXPECTED to fail and that FAIL is the finding.
#
#   8 mma_switch     STUDY item 3's loose end, cheap: is the instruction chosen
#                    by the tile alone, at fixed tokens.
#   9 ruler          Prices a ridge change without making one. Last of the
#                    ridge-related arms because it changes how none above is
#                    read: --write-calibration is off.
#  10 cap_test       BLOCK_M=16, DEMOTED. vLLM runs 16 multi-tile in 1 of 24
#                    cells, so this tests the FORMULA, not production.
#  11 dtype          Is the 1.15 fp8/bf16 crossing the FORMAT or the CONFIG.
#  12 span_dense     Is the 0.563 the span EXTENT or the KERNEL. The DENSE grid
#                    runs first: it is the only grid on which C3's mechanism is
#                    observable at all, and the sparse grid's own kernel world
#                    predicts C2 FAIL. A CLAIM gate failing there is a RESULT
#                    (exit 1, CLAIM_FAIL), never a retry.
#  12 span           The sparse grid, second, for the extent comparison.
#  13 counter_plan   Free. Probes whether a DRAM counter route is open on this
#                    box and prints the manual ncu command if so. A counter is
#                    the ONLY thing that turns alpha_b into a number rather than
#                    an interval. THIS LINE SAID "AND IT IS BLOCKED ON RENTED
#                    PODS" UNTIL 2026-09-10: it read OPEN on the 2026-09-09 and
#                    2026-09-10 RunPod H200s, which is why arms 14 and 15
#                    exist.
#  14 counter-n32-m64   THE COUNTER RUN, added 2026-09-10 because the probe
#  15 counter-n128-m64  above came back OPEN for the second rental running.
#                    They are LAST and they are 120 WALL minutes EACH. Reading
#                    DRAM traffic at TWO BLOCK_N values at fixed BLOCK_M is the
#                    only contrast that separates a per-N-tile TRAFFIC cost from
#                    a per-N-tile TIME cost, and a contrast needs two cells, so
#                    it is two arms and not one: --block-m 64 on both,
#                    --block-n 32 and 128, and the BLOCK_M is on the arm lines
#                    because the pre-registered discriminator is registered
#                    there. UNTIL 2026-09-10 THIS WAS ONE ARM PASSING NEITHER
#                    FLAG, which ran the script's default BLOCK_N=64 BLOCK_M=32
#                    cell while three places in this file and one in the runbook
#                    said it ran the contrast. Either arm on its own still buys
#                    the other half: alpha_b as a traffic slope with no fitted
#                    level, no delta and no assumed bandwidth. They run last
#                    because every arm above is cheaper and because a BLOCKED
#                    probe retires both for the whole session at no cost.
#
# EVERY ARM IS INDEPENDENT AND NONE IS FATAL. One failing arm records its status
# and the rest continue; a long unattended run that aborts at minute six on a
# shape that does not compile is worse than no automation.
#
# THE STATES. Measuring, straight from moe/bench/exit_codes.py:
#
#   DONE        0  measured, every VALIDITY and CLAIM gate PASSED
#   CLAIM_FAIL  1  measured, VALIDITY passed, a CLAIM gate did not. A RESULT.
#                  The arm is FINISHED and is never re-run: repeating it spends
#                  the same minutes to obtain the same refutation.
#   REFUSED     2  nothing measured, a precondition was not met. Free. A re-run
#                  of this driver DOES re-attempt it, because a refusal costs no
#                  pod minutes and the usual reason to re-run is that the
#                  precondition was fixed.
#   INVALID     3  measured, then a VALIDITY gate FAILED. Nothing on the page
#                  may be quoted, and the arm is NOT auto-retried: the cause is
#                  the instrument, and repeating it repeats the failure unless
#                  the log says in words that it was transient.
#   RETRY     4+  a code nobody chose. Read the log.
#
# Planning (--dry-run), which is a different question and gets different words:
#
#   PLANNED       the arm printed its plan (rc 0)
#   PLAN_REFUSED  the arm refused to plan (rc 2), which off a GPU box is the
#                 refusal working and costs nothing
#   BROKEN        anything else, a traceback included. The plan on the page is
#                 not a plan, and the session exits non-zero.
#   NOT_PLANNED   the arm has no plan mode and was not run. It is named with the
#                 reason instead of being silently absent.
#
# WHAT IT NEVER DOES. It never terminates a pod, never commits and never pushes.
# It prints what to commit and stops. It also never writes into the work tree:
# the anchor re-score is given an --out-dir under the session directory, because
# its default is results/published/ and it used to rewrite two TRACKED files on
# every --dry-run.
#
# THREE SHELL HABITS THIS FILE AVOIDS ON PURPOSE, each of which has already cost
# this project a run:
#   * NO `set -e`. See above.
#   * NO PIPELINE around a measured command. `cmd > log 2>&1` then read the log,
#     never `cmd | tee log`, because a pipeline reports the exit status of its
#     LAST element and the ledger would record the success of `tee`.
#   * NO BACKGROUND JOB, therefore no `$!` and no PID variable read in a shell
#     that started nothing.
set -uo pipefail          # NOT -e: an arm may fail and the run must continue

# --------------------------------------------------------------------------
# Where things are. REPO is derived from this file rather than hardcoded to
# /workspace/repo, so --dry-run works from a checkout on a laptop.
# --------------------------------------------------------------------------
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY_BASE="${PY_BASE:-/workspace/venvs/base/bin/python}"
PY_VLLM="${PY_VLLM:-/workspace/venvs/vllm/bin/python}"
[[ -x "$PY_BASE" ]] || PY_BASE="$REPO/.venv/bin/python"
[[ -x "$PY_BASE" ]] || PY_BASE="$(command -v python3 || true)"
[[ -x "$PY_VLLM" ]] || PY_VLLM="$PY_BASE"

# The driver's own exit codes speak the same table as the arms'. A stop before
# anything is measured is REFUSED (2); a plan that did not survive its own
# --dry-run is INVALID (3), because what is on the page is not a plan.
RC_REFUSED=2
RC_INVALID=3
RC_RETRY=4

DRY=0
ONLY=""
LIST=0
RESUME_LATEST=0
NEW_SESSION=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --list)    LIST=1; shift ;;
    --only)    ONLY="$2"; shift 2 ;;
    --resume-latest) RESUME_LATEST=1; shift ;;
    --new)     NEW_SESSION=1; shift ;;
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' \
                 "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "REFUSED: unknown argument: $1" >&2; exit "$RC_REFUSED" ;;
  esac
done

# >>> LIFTABLE: function definitions only, no top-level statements.
# tests/test_h200_gaps_session.py evaluates everything between these two markers
# in its own shell and calls the functions with planted inputs, because the
# states below are exactly what cannot be checked by reading a pod log an hour
# later. Nothing in here may run at source time or the test would run the
# session.

say()  { printf '\n==== %s ====\n' "$*"; }
note() { printf '  %s\n' "$*"; }

# THE SHELL MIRROR OF moe/bench/exit_codes.ledger_state. Kept here rather than
# shelling out to python once per arm, and pinned by a test that compares this
# function with that function for every code from 0 to 6 plus 127 and 130. If
# the table ever changes, that test fails before a pod does.
ledger_state() { case "$1" in
  0) echo DONE ;;
  1) echo CLAIM_FAIL ;;
  2) echo REFUSED ;;
  3) echo INVALID ;;
  *) echo RETRY ;;
esac; }

# WHAT THE LOG SAYS THE EXIT CODE SHOULD HAVE BEEN. One of three answers, and
# the caller must handle all three: an integer, which is exit_codes.classify_text
# over the RESULT lines the arm printed; NONE, when the log carries no RESULT
# line at all, which is what a refusal looks like from here and what a crash
# before the first gate looks like too; UNREADABLE, when the second opinion
# could not be formed (no log, no interpreter, a moe/bench/exit_codes that
# would not import). UNREADABLE is not NONE: "could not check" is a different
# answer from "checked and found nothing", and `second_opinion` refuses to latch
# on either. Shells out to $PY_BASE once per arm, which is seconds against
# minutes, and asks the module itself rather than a shell re-implementation of
# its regex, so the driver and the scripts read one line format from one file.
log_verdict() {
  local log="${1:-}" out rc=0
  [[ -n "$log" && -f "$log" ]] || { echo UNREADABLE; return 0; }
  out="$("$PY_BASE" - "$REPO" "$log" <<'PY' 2>/dev/null
import sys
from pathlib import Path

repo, log = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
try:
    from moe.bench import exit_codes as EC
except Exception as exc:                                          # noqa: BLE001
    print(f"UNREADABLE {exc.__class__.__name__}")
    raise SystemExit(0)
text = Path(log).read_text(errors="replace")
try:
    print(EC.classify_text(text))
except EC.NoGatesScored:
    print("NONE")
PY
)" || rc=$?
  if (( rc != 0 )) || [[ -z "$out" ]]; then echo UNREADABLE; return 0; fi
  printf '%s\n' "$out"
}

# THE SECOND OPINION, TAKEN. $1 the exit code, $2 the word `ledger_state` gave
# it, $3 what `log_verdict` read off the log, $4 the rc of `adopts_exit_codes`
# for the file the arm ran (0 adopted, 1 not, 2 could not be found). Prints one
# line, `<state>TAB<note>`, and the note goes into the ledger row so the reason
# a row is not latched travels with the row.
#
# WHY THIS EXISTS. exit_codes.classify_text was written on 2026-09-02 so that
# "the exit code a log implies can be recomputed from its RESULT lines and
# compared with the code the process returned, and a disagreement is itself a
# defect". Until 2026-09-03 nothing in this file called it: `arm` read the
# integer and `summarize_arm` grepped the lines for display. Every rule below
# was proven, not argued, with this file's own LIFTABLE `arm()` over planted
# commands before it was written:
#
#   RESULT lines present, code in the table, and they AGREE   the word stands
#   RESULT lines present, code in the table, and they DIFFER  DEFECT: UNKNOWN.
#       A `RESULT: CLAIM C1 FAIL` page under exit 0 was latched DONE. Both
#       codes go in the note; the closing summary prints these rows under their
#       own heading; the session exits INVALID over them, because a page that
#       contradicts its own exit code is not a verdict, in either direction.
#   RESULT lines present, code OUTSIDE the table                RETRY, and NOT a
#       defect: the arm printed its gates and then crashed (ERROR 4, a signal,
#       a 127). The process code wins, the note says what the page implied.
#   no RESULT line, exit 0                                       UNEARNED DONE:
#       UNKNOWN. A check that examined nothing reports no failures, and DONE
#       is the word this ledger latches hardest.
#   no RESULT line, exit 1, file speaks the table                CRASH: RETRY.
#       An adopting script prints one RESULT line per scored gate and exits
#       through `classify`, and moe/bench/cli.py registers only VALIDITY gates
#       and cannot return 1 on purpose; so a 1 with no gate line is an
#       exception that escaped BEFORE any gate, which the ERROR(4) guards
#       around `_main()` cannot catch: an import that drifted, a module-level
#       failure. The pod's real 2026-09-01 shape. scripts/check_mma_path.sh
#       reads exactly this signal for its own child and maps it to ERROR.
#   no RESULT line, exit 1, file does not speak the table        UNKNOWN, as the
#       third pass already had it: from such a file 1 is three things at once.
#   no RESULT line, exit 2                                       REFUSED. The
#       expected shape of a refusal, and the one no-gate exit that is earned.
#   no RESULT line, exit 3                                       UNEARNED
#       INVALID: UNKNOWN. INVALID names a VALIDITY gate that failed, and none
#       was scored. Not latched.
#   UNREADABLE                                                   UNKNOWN. The
#       driver could not form the opinion, so it cannot vouch for any word.
#
# Never returns non-zero and never prints more than one line, because `arm`
# reads it with one `read`.
second_opinion() {
  local rc="$1" state="$2" implied="$3" adopts="$4"
  case "$implied" in
    UNREADABLE*)
      printf 'UNKNOWN\tSECOND OPINION UNAVAILABLE: exit %s, but exit_codes.classify_text could not be run over the log (%s). A word this driver could not check is not latched.\n' \
        "$rc" "$implied"
      return 0 ;;
  esac
  if [[ "$state" == RETRY ]]; then
    if [[ "$implied" == NONE ]]; then
      printf 'RETRY\t\n'
    else
      printf 'RETRY\texit %s is outside the table while the RESULT lines imply %s %s: the arm printed its gates and then did not exit through them. The process code wins; read the traceback.\n' \
        "$rc" "$implied" "$(ledger_state "$implied")"
    fi
    return 0
  fi
  if [[ "$implied" == NONE ]]; then
    case "$rc" in
      0) printf 'UNKNOWN\tUNEARNED DONE: exit 0 with no RESULT line. A check that examined nothing reports no failures. NOT latched.\n' ;;
      1) if (( adopts == 0 )); then
           printf 'RETRY\tCRASH: exit 1 with no RESULT line from a file that speaks the table. An adopting script scores a gate before it can exit 1, so this is an exception that escaped before any gate (an import that drifted, a module-level failure), not a refuted claim. Not latched; read the tail of the log.\n'
         else
           printf 'UNKNOWN\texit 1 with no RESULT line from a file that has not adopted moe/bench/exit_codes, where 1 is three things at once. NOT latched.\n'
         fi ;;
      2) printf 'REFUSED\t\n' ;;
      3) printf 'UNKNOWN\tUNEARNED INVALID: exit 3 with no RESULT line. INVALID names a VALIDITY gate that failed and none was scored. NOT latched.\n' ;;
      *) printf 'UNKNOWN\texit %s with no RESULT line and no rule for that pair. NOT latched.\n' "$rc" ;;
    esac
    return 0
  fi
  if [[ "$implied" != "$rc" ]]; then
    printf 'UNKNOWN\tDEFECT: the process exited %s %s but its RESULT lines imply %s %s. The page and the exit code disagree, which moe/bench/exit_codes names as itself a defect. NOT latched.\n' \
      "$rc" "$(ledger_state "$rc")" "$implied" "$(ledger_state "$implied")"
    return 0
  fi
  if [[ "$rc" == 1 ]] && (( adopts != 0 )); then
    printf 'UNKNOWN\texit 1 from a file that has not adopted moe/bench/exit_codes; its RESULT lines agree with 1, but the file never agreed to the table they are read by. NOT latched.\n'
    return 0
  fi
  printf '%s\tlog agrees: RESULT lines imply %s\n' "$state" "$implied"
}

# THE ROWS OF A LEDGER WHOSE PAGE CONTRADICTS THEIR EXIT CODE, one line each,
# last row per arm winning as everywhere else in this file, so a resume that
# did not touch a defective arm still shows it and a resume that re-ran it
# cleanly does not. Empty output is the all-clear; the caller decides what to
# print for it. Read off the ledger rather than counted in a global so the
# summary and the exit code have one source.
defect_rows() {
  awk -F'\t' 'NR > 1 { st[$1] = $2; rc[$1] = $3; nt[$1] = $7
                       if (!($1 in seen)) { order[++n] = $1; seen[$1] = 1 } }
              END { for (i = 1; i <= n; i++) { a = order[i]
                      if (st[a] == "UNKNOWN" && nt[a] ~ /^DEFECT:/)
                        printf "  %-19s exit %s   %s\n", a, rc[a], nt[a] } }' "$1"
}

# THE ROWS THIS SESSION STILL OWES: every UNKNOWN row that is NOT a DEFECT,
# same last-row-wins rule. An UNEARNED DONE (exit 0, no RESULT line), an
# UNEARNED INVALID, an exit 1 from a non-adopting file, a log the second
# opinion could not read: each is an arm that examined nothing the driver could
# vouch for, is not latched, and re-runs on --resume-latest. Until 2026-09-08
# the session's exit code reached INVALID only through `defect_rows`, so a
# ledger holding such a row exited 0, the runbook's next line was the exfil
# tar, and nothing machine-readable said an arm was still owed. The exit rule
# is `session_rc` below, over both lists, so the summary and the code have one
# source and both are plantable.
owed_rows() {
  awk -F'\t' 'NR > 1 { st[$1] = $2; rc[$1] = $3; nt[$1] = $7
                       if (!($1 in seen)) { order[++n] = $1; seen[$1] = 1 } }
              END { for (i = 1; i <= n; i++) { a = order[i]
                      if (st[a] == "UNKNOWN" && nt[a] !~ /^DEFECT:/)
                        printf "  %-19s exit %s   %s\n", a, rc[a], nt[a] } }' "$1"
}

# THE SESSION'S OWN EXIT CODE, from the ledger and the crash count, printed:
#   RC_INVALID  any UNKNOWN row at all, DEFECT or owed. Nothing on such a page
#               is a verdict, and exit 0 was read as "every arm produced a
#               result" by whoever runs the exfil line next.
#   RC_RETRY    otherwise, when any arm this pass exited a code outside the
#               table or crashed before scoring a gate.
#   0           otherwise. It still does not mean every arm is DONE: CLAIM_FAIL
#               and INVALID are results and REFUSED is free; read the ledger.
# The two constants are read from the environment on purpose, so a lifted test
# that forgets to pass them fails unbound rather than reading a default.
session_rc() {
  local ledger="$1" retry="${2:-0}"
  if [[ -n "$(defect_rows "$ledger")" || -n "$(owed_rows "$ledger")" ]]; then
    echo "$RC_INVALID"; return 0
  fi
  if (( retry > 0 )); then echo "$RC_RETRY"; return 0; fi
  echo 0
}

# THE NEWEST SESSION FOR THIS CARD THAT HOLDS A MEASURING LEDGER, or rc 1 and
# nothing. "Newest" is by name, and the names carry a UTC stamp, so by name is
# by time. A directory with no ARMS.tsv is not a session to resume into: a
# --dry-run leaves ARMS-dryrun.tsv only, and that ledger latches nothing.
latest_session() {
  local root="$1" prefix="$2" d found=""
  for d in "$root/$prefix"*/; do
    [[ -f "$d/ARMS.tsv" ]] && found="${d%/}"
  done
  [[ -n "$found" ]] || return 1
  printf '%s\n' "$found"
}

# WHICH DIRECTORY THIS SESSION RUNS IN, decided in one place so that every
# branch can be planted. $1 DRY, $2 --resume-latest, $3 --new, $4 SESSION= as
# given (empty when not), $5 the session root, $6 the per-card prefix. Prints
# `<how>TAB<value>` and returns 1 on the two refusals:
#   NAMED     <dir>    SESSION= named it; the flags may not also be given
#   RESUMED   <dir>    --resume-latest, and a session with ARMS.tsv exists
#   NEW       <dir>    a fresh stamped directory
#   REFUSED   <why>    contradictory flags, or nothing to resume
#   LATEST_EXISTS <dir> a measuring run without --new found a session it would
#                      have silently ignored. The caller words the refusal.
#
# WHY A MEASURING RUN REFUSES AND A DRY RUN DOES NOT. The latch, "a DONE or
# CLAIM_FAIL or INVALID row is never re-run", lives in ARMS.tsv and nowhere
# else. A fresh directory has an empty ledger, so a plain re-invocation
# re-attempted every finished arm and re-ran the rows the latch exists to
# protect, and nothing in --help said SESSION= was how to avoid that. A dry run
# skips nothing and spends nothing, so a fresh directory costs it nothing.
session_choice() {
  local dry="$1" resume="$2" new="$3" explicit="$4" root="$5" prefix="$6" latest=""
  if (( resume )) && (( new )); then
    printf 'REFUSED\t--resume-latest and --new contradict each other. Give one.\n'
    return 1
  fi
  if [[ -n "$explicit" ]]; then
    if (( resume )) || (( new )); then
      printf 'REFUSED\tSESSION=%s names the directory outright, and --resume-latest / --new choose one under %s. Give one or the other.\n' \
        "$explicit" "$root"
      return 1
    fi
    printf 'NAMED\t%s\n' "$explicit"
    return 0
  fi
  latest="$(latest_session "$root" "$prefix")" || latest=""
  if (( resume )); then
    if [[ -z "$latest" ]]; then
      printf 'REFUSED\t--resume-latest found no session with a measuring ledger under %s/%s*. Nothing to resume; run without it to open one.\n' \
        "$root" "$prefix"
      return 1
    fi
    printf 'RESUMED\t%s\n' "$latest"
    return 0
  fi
  if (( dry == 0 )) && (( new == 0 )) && [[ -n "$latest" ]]; then
    printf 'LATEST_EXISTS\t%s\n' "$latest"
    return 1
  fi
  printf 'NEW\t%s/%s%s\n' "$root" "$prefix" "$(date -u +%Y%m%dT%H%M%SZ)"
}

# WHICH FILE EACH ARM ACTUALLY RUNS, repo-relative, so this driver can ASK that
# file whether it speaks the table above instead of assuming it does. It decides
# NO arm's state: R1 deleted the per-arm done-code lists and this restores none.
# `ledger_state` is still the only thing that turns an exit code into a word.
arm_script() { case "$1" in
  calibrate)                     echo scripts/calibrate_hardware.py ;;
  pin_probe-*)                   echo moe/bench/cli.py ;;
  roofline-*)                    echo scripts/bm128_roofline.py ;;
  bm128_depth)                   echo scripts/bm128_depth.py ;;
  alias_ablation)                echo scripts/alias_ablation.py ;;
  noise_floor)                   echo scripts/replicate_noise_floor.py ;;
  bn_g16)                        echo scripts/bn_decomposition.py ;;
  anchor_measure|anchor_rescore) echo scripts/memory_branch_anchor.py ;;
  occupancy)                     echo scripts/occupancy_vs_swizzle.py ;;
  mma_switch)                    echo scripts/check_mma_path.sh ;;
  ruler)                         echo scripts/ruler_rebaseline.py ;;
  cap_test)                      echo scripts/tile_cap_test.py ;;
  dtype)                         echo scripts/dtype_tile_confound.py ;;
  span|span_dense)               echo scripts/span_extent_separation.py ;;
  counter_plan|counter_contrast|counter-*)
                                 echo scripts/dram_counter_route.py ;;
  *)                             echo "" ;;
esac; }

# Does the file an arm runs speak moe/bench/exit_codes' table. ASKED of the file
# on every run rather than carried in a list here, because a list of who has
# adopted goes stale the day someone adopts, which is the failure mode R1
# removed. rc 0 adopted, rc 1 not, rc 2 UNKNOWN -- a file this cannot find, which
# is not the same answer as adopted and is treated here as not.
adopts_exit_codes() {
  local rel="$1"
  [[ -n "$rel" && -f "${REPO:-}/$rel" ]] || return 2
  grep -q 'moe\.bench\.exit_codes\|moe\.bench import exit_codes' -- "${REPO:-}/$rel"
}

# WHAT A ROW MAY ACTUALLY MEAN when the file that produced it has not adopted
# the table this session reads it by. Prints nothing -- rc 1 -- for a file that
# has adopted, so it is silent on every row whose word is trustworthy. It
# changes no state and no exit code; the whole content is disclosure, which is
# what the measuring path did not have.
#
# IT USED TO COVER TWO STATES AND THE TWO NON-ADOPTERS SPENT NEITHER. REFUSED
# and INVALID were the only branches, while scripts/check_mma_path.sh spent 1
# on a failed gate and moe/bench/cli.py spent 4 on one, so a ledger holding
# nothing but `mma_switch CLAIM_FAIL 1` walked every row, matched none, and
# printed the ALL-CLEAR over the single row whose word was wrong. Every state a
# non-adopting file can produce is covered here now, and the all-clear says
# "in any state" rather than naming two.
contract_caveat() {
  local name="$1" state="$2" rel why rc=0
  rel="$(arm_script "$name")"
  adopts_exit_codes "$rel" || rc=$?
  case "$rc" in
    0) return 1 ;;
    2) why="which this driver could not find, so whether it speaks that\n  module is UNKNOWN -- and UNKNOWN is not the same answer as adopted" ;;
    *) why="which does not import moe/bench/exit_codes" ;;
  esac
  case "$state" in
    REFUSED)
      printf '  CAVEAT: this row came from %s,\n' "${rel:-a command outside scripts/}"
      printf '  %b.\n' "$why"
      printf '  It therefore reads REFUSED for one reason and one only: the command\n'
      printf '  exited 2. A file that has not adopted the table may spend 2 on\n'
      printf '  something else. scripts/memory_branch_anchor.py used to document 2\n'
      printf '  as a VALIDITY gate that failed AFTER an eight-minute measurement,\n'
      printf '  the opposite reading, under which such a run is unquotable rather\n'
      printf '  than free; it has since adopted the table, so it no longer reaches\n'
      printf '  this caveat, and that is the shape of the risk for whatever has not.\n'
      printf '  Read the log before believing that nothing was measured, and\n'
      printf '  before re-running it.\n' ;;
    INVALID)
      printf '  CAVEAT: this row came from %s,\n' "${rel:-a command outside scripts/}"
      printf '  %b.\n' "$why"
      printf '  It therefore reads INVALID for one reason and one only: the command\n'
      printf '  exited 3. A file that has not adopted the table may spend 3 on a\n'
      printf '  REFUSAL, which measured nothing and costs nothing to re-run, or on\n'
      printf '  a registered ANSWER: scripts/dram_counter_route.py used to return 3 for\n'
      printf '  every verdict that was not OPEN, and BLOCKED on a rented pod is what\n'
      printf '  that arm exists to find out, not a broken instrument; since adopting\n'
      printf '  the table on 2026-09-02 it scores one gate per verdict (OPEN DONE,\n'
      printf '  BLOCKED CLAIM_FAIL, no ncu on PATH REFUSED) and no longer reaches\n'
      printf '  this caveat. INVALID rows are latched and skipped on every\n'
      printf '  later run; delete this row from the ledger to run the arm again.\n' ;;
    CLAIM_FAIL|UNKNOWN)
      printf '  CAVEAT: this row came from %s,\n' "${rel:-a command outside scripts/}"
      printf '  %b.\n' "$why"
      printf '  The command exited %s; the ledger note reads: %s\n' \
        "$(ledger_arm_rc "$name")" "$(ledger_arm_note "$name")"
      printf '  Under the table 1 is CLAIM_FAIL: measured,\n'
      printf '  VALIDITY passed, a pre-registered CLAIM did not -- a RESULT, and the\n'
      printf '  one state this ledger LATCHES as finished so the arm is never spent\n'
      printf '  again. From a file that has not adopted the table, 1 is three things\n'
      printf '  at once. scripts/check_mma_path.sh used to document "1 a gate failed"\n'
      printf '  and spend it on the instruction-follows-the-tile reading, which is a\n'
      printf '  VALIDITY gate, AND on "no interpreter at $PY" and "no .ptx under the\n'
      printf '  dump dir", which measured nothing and are refusals; since adopting\n'
      printf '  the table it spends 2 on those refusals and no longer reaches this\n'
      printf '  caveat, and that is the shape of the risk for whatever has not. And\n'
      printf '  1 is what Python returns for any exception that escapes main, which\n'
      printf '  is ERROR.\n'
      printf '  So this driver records UNKNOWN and does NOT latch the row: a state\n'
      printf '  it cannot tell apart is not a result it may file. Read the log --\n'
      printf '  the three cases do not resemble each other in it -- and delete or\n'
      printf '  keep the row deliberately.\n' ;;
    DONE)
      printf '  CAVEAT: this row came from %s,\n' "${rel:-a command outside scripts/}"
      printf '  %b.\n' "$why"
      printf '  It therefore reads DONE for one reason and one only: the command\n'
      printf '  exited 0. Under the table 0 means every VALIDITY and every CLAIM\n'
      printf '  gate PASSED. A file that has not adopted it may spend 0 on a run\n'
      printf '  that scored no gate at all: scripts/check_mma_path.sh used to exit 0\n'
      printf '  from its own --dry-run, and again from a ladder path whose closing\n'
      printf '  lines said the cell could not attribute the instruction to a tile;\n'
      printf '  since adopting the table its --dry-run exits 2 REFUSED and it no\n'
      printf '  longer reaches this caveat.\n'
      printf '  DONE is latched and skipped on every later run, so read the RESULT\n'
      printf '  lines above before taking this row for gates that passed. A check\n'
      printf '  that examined nothing reports no failures.\n' ;;
    RETRY)
      printf '  CAVEAT: this row came from %s,\n' "${rel:-a command outside scripts/}"
      printf '  %b.\n' "$why"
      printf '  It therefore reads RETRY only because the command exited a code the\n'
      printf '  table does not name, and this session exits 4 over it and the next\n'
      printf '  one attempts the arm again. A file that has not adopted the table\n'
      printf '  may spend such a code on a REGISTERED ANSWER rather than a crash:\n'
      printf '  moe/bench/cli.py used to return 4 when the implementations ran under\n'
      printf '  the pin and no row showed that tile, which is the VALIDITY failure\n'
      printf '  that probe exists to detect, not an unplanned exception; it now\n'
      printf '  scores F1 and F2 through exit_codes.classify, exits 3 INVALID there,\n'
      printf '  and no longer reaches this caveat. Re-running such an arm spends the\n'
      printf '  minutes again to reach the same number. Read the log before the\n'
      printf '  next session does.\n' ;;
    *) return 1 ;;
  esac
  return 0
}

# THE MEASURING PATH'S COUNTERPART TO THE DRY-RUN BANNER, which said only under
# --dry-run, where it costs nothing, that a refusal exiting 3 is a refusal
# wearing an INVALID's number. On the pod the same collision costs an arm and
# nothing said it. Reads the ledger rather than the arm list so it can be tested
# with a planted ledger, and so an arm added later is covered without being
# named twice. Both branches print: the all-clear is a sentence, not silence,
# because an empty section reads as nothing to report.
contract_disclosure() {
  local ledger="$1" n state rest rel any=0
  while IFS=$'\t' read -r n state rest; do
    # EVERY STATE A NON-ADOPTING FILE CAN PRODUCE, not the two this used to
    # name. The header row and NOT_PLANNED fall out here because neither is a
    # state a command exited with.
    case "$state" in
      DONE|CLAIM_FAIL|REFUSED|INVALID|RETRY|UNKNOWN) ;;
      *) continue ;;
    esac
    rel="$(arm_script "$n")"
    [[ -n "$rel" ]] && adopts_exit_codes "$rel" && continue
    if (( any == 0 )); then
      say "THE ROWS WHOSE STATE MAY BE THE WRONG WORD"
      printf '  This session reads every exit code through moe/bench/exit_codes.\n'
      printf '  The files below have not adopted that module, so their codes were\n'
      printf '  chosen against some other table and the state in the ledger is a\n'
      printf '  translation nobody agreed to. Nothing here is patched per arm; the\n'
      printf '  fix belongs in those files, through exit_codes.classify.\n\n'
      any=1
    fi
    printf '  %-19s %s   from %s\n' "$n" "$state" "${rel:-a command outside scripts/}"
    contract_caveat "$n" "$state"
    printf '\n'
  done < "$ledger"
  if (( any == 0 )); then
    say "THE EXIT-CODE CONTRACT"
    printf '  No row in this session, in ANY state and not only REFUSED and\n'
    printf '  INVALID, came from a file that has not adopted moe/bench/exit_codes,\n'
    printf '  so every state word above is the one that table defines.\n'
  fi
  return 0
}

# A PLAN IS NOT A MEASUREMENT, so a plan does not get measuring words. Three
# outcomes, and only one of them is silent-failure-shaped: a --dry-run that
# tracebacks exits 1, which lands in BROKEN and stops the session, which is the
# defect this mapping exists to catch.
#
# rc 2 IS NOT A WORD ON ITS OWN, and that is the 2026-09-02 regression. When
# every script adopted the table, every one of their --dry-runs began exiting
# REFUSED -- correctly: a plan measured nothing and scored no gate -- so eleven
# of the sixteen planned arms landed on PLAN_REFUSED and DRY mode lost the
# only distinction it exists to draw. The log still carries it, so rc 2 is
# decided by `printed_a_plan` and not by the number: a plan that printed and
# then refused is PLANNED, a refusal with nothing printed before it is
# PLAN_REFUSED. Called with no log, rc 2 is PLAN_REFUSED: nothing printed.
dry_state() { case "$1" in
  0) echo PLANNED ;;
  2) if printed_a_plan "${2:-}"; then echo PLANNED; else echo PLAN_REFUSED; fi ;;
  *) echo BROKEN ;;
esac; }

# The P1 check, re-asked after every arm rather than once at preflight. Two arms
# used to write into the tracked tree while the session ran -- calibrate_hardware
# into moe/bench/hardware/measured_<card>.yaml and the anchor --rescore into
# results/published/ANCHOR_RESCORE.{txt,json} -- so "0 dirty files" at preflight
# was true and meaningless, and 44,872 published rows carry git_dirty=True.
# --untracked-files=all counts FILES, matching moe/bench/provenance.py. Prints
# "-" when git cannot answer, which is not the same number as 0.
dirty_count() {
  local out rc=0
  out="$(git -C "$REPO" status --porcelain --untracked-files=all 2>/dev/null)" || rc=$?
  if (( rc != 0 )); then echo "-"; return 0; fi
  if [[ -z "$out" ]]; then echo 0; else printf '%s\n' "$out" | wc -l | tr -d ' '; fi
}

# AN ISO-8601 UTC STAMP AS THE INTEGER YYYYMMDDhhmmss, or nothing at all when
# the string does not carry one. Compared as an integer rather than through
# `date -d`, which is GNU-only and absent on the laptop half of this project,
# and rather than with `[[ a > b ]]`, which collates by locale: which machine
# reads the clock must not decide whether a session runs. Returns 1 on a string
# it cannot read, so no caller can mistake "no timestamp" for "an old one".
utc_stamp() {
  local raw="${1:-}" digits
  raw="${raw%%+*}"
  digits="$(printf '%s' "$raw" | tr -cd '0-9')"
  (( ${#digits} >= 14 )) || return 1
  printf '%s\n' "${digits:0:14}"
}

# WHEN THE TRACKED CALIBRATION SAYS IT WAS MEASURED, as that integer.
# calibrate_hardware.py writes moe/bench/provenance.provenance_block under a
# top-level `provenance:` key, and `utc` is the one field in it that answers
# "was this THIS rental". Read with awk over that block alone: `detail:` carries
# a hundred nested keys and a grep for `utc:` over the whole file would find
# whichever of them came first. rc 2 no such file, rc 3 a file with no stamp.
calibration_stamp() {
  local yaml="$1" raw
  [[ -f "$yaml" ]] || return 2
  raw="$(awk '/^provenance:/ {p = 1; next}
              p && /^[^[:space:]]/ {p = 0}
              p && $1 == "utc:" {print $2; exit}' "$yaml" | tr -d "\"'")"
  [[ -n "$raw" ]] || return 3
  utc_stamp "$raw"
}

# WHOSE RULER THE ARMS BELOW WILL BE SCORED AGAINST. One word, so both branches
# are plantable with a yaml and a stamp and neither is reachable only on a pod:
#   PUBLISHED   the tracked file for this card exists and carries a stamp at or
#               after this session's own start. Arm 0 measured THIS card and
#               published it, which is the only state the rest may run in.
#   MISSING     no tracked calibration for this card at all: every arm that
#               needs a ridge would refuse, one at a time, for the whole
#               session.
#   UNDATED     a tracked file with no provenance.utc. It cannot say which
#               rental measured it, and "cannot say" is not "this one". THIS
#               LINE NAMED THE COMMITTED measured_nvidia_h200.yaml AS AN
#               EXAMPLE UNTIL 2026-09-10, and it had stopped being one: the
#               2026-09-09 calibration added the provenance block and the
#               2026-09-10 one carries utc 2026-09-10T04:12:29+00:00, so the
#               committed ruler is decided on its DATE and never on "cannot
#               say". The state is still reachable and is planted in
#               tests/test_h200_gaps_session.py rather than read off a
#               tracked file that has moved on.
#   STALE       a tracked file measured BEFORE this session began -- the
#               previous rental's, still in the checkout. This is the state the
#               A6 fix created: arm 0 measures this card into an untracked
#               session path and every later arm quotes the last pod's ridge
#               while `ridge_source` names the attached device. The audit's own
#               "a constant from another machine presented as a measurement".
#   NO_BASELINE this driver could not stamp its own start, so it cannot say
#               which side of it the file is on. Refuse rather than guess.
calibration_state() {
  local yaml="$1" since="${2:-}" stamp rc=0
  [[ -n "$since" ]] || { echo NO_BASELINE; return 0; }
  stamp="$(calibration_stamp "$yaml")" || rc=$?
  case "$rc" in
    0) ;;
    2) echo MISSING;  return 0 ;;
    *) echo UNDATED;  return 0 ;;
  esac
  if (( 10#$stamp >= 10#$since )); then echo PUBLISHED; else echo STALE; fi
}

# THE CALIBRATION GATE'S DECISION, from the two words that answer its two
# questions, in ONE place so that the session and its test read the same rule.
# Prints the half that failed and the word that failed it -- "ARM INVALID",
# "YAML STALE" -- or OK, and returns non-zero on anything but OK. A missing
# calibrate row prints ARM NO_ROW rather than ARM, because an empty word in a
# refusal reads as a bug in the refusal.
#
# WHY THE ARM HALF COMES FIRST. When --only leaves arm 0 out, the yaml half
# reports on a file this session never touched, and its answer (UNDATED, on the
# committed H200 ruler) names the wrong problem. The arm half names --only.
calibration_verdict() {
  local row="$1" state="$2"
  [[ "$row" == DONE ]]      || { echo "ARM ${row:-NO_ROW}"; return 1; }
  [[ "$state" == PUBLISHED ]] || { echo "YAML $state";      return 1; }
  echo OK
}

# THE GRADE OF THE REFERENCE CLOCK THE PUBLISHED RULER CARRIES, one line:
#   <grade>|<mhz>|<usable_for_roof>|<source>
# read through roofline.reference_clock over the card's tracked yaml, which is
# the same resolver every arm levels against. `grade` is HOW the number was
# taken (roofline.REFERENCE_UNDER_LOAD, _IDLE_SCALAR, _SETTLE_PLATEAU, or NONE
# when the file carries no clock) and `usable_for_roof` is True only for the
# under-load median. UNREADABLE when the module cannot be imported, which is
# reported and refused rather than read as a grade. No apostrophes in the
# heredoc: it sits inside a command substitution on a bash-3.2 laptop.
reference_grade() {
  local card="$1" dir="$2"
  "$PY_BASE" - "$REPO" "$card" "$dir" <<'PY' 2>/dev/null
import sys
from pathlib import Path
repo = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo))
try:
    from moe.bench import roofline
except Exception as exc:                                          # noqa: BLE001
    print(f"UNREADABLE|0|False|moe.bench.roofline is not importable from {repo}: "
          f"{exc.__class__.__name__}: {exc}")
    raise SystemExit(0)
ref = roofline.reference_clock(sys.argv[2], directory=Path(sys.argv[3]))
mhz = int(ref.mhz) if ref.mhz else 0
print(f"{ref.grade or 'NONE'}|{mhz}|{ref.usable_for_roof}|{ref.source}".replace(chr(10), " "))
PY
}

# THE REFUSAL FOR A RULER WHOSE CLOCK THE ROOF CANNOT BE RESCALED AGAINST.
# WHY THIS GATE EXISTS. The calibration gate above asks WHEN the yaml was
# written and whether arm 0 stood behind it. It does not ask what KIND of clock
# the yaml carries, and every arm below scores LEVEL against that clock and
# writes a per-row roof (roof_at_cell_clock_tflops, pct_of_roof_at_cell_clock)
# from it. roofline.reference_clock grades the field it read: only the median
# sampled WHILE the calibration GEMM ran (under-load) may rescale the roof; the
# post-hoc idle scalar, which calibrate.py records moving 30% across eleven
# calibrations of one card, and the settle plateau are DISOWNED, so against
# them the driver refuses the per-row roof on every row and LEVEL is
# provisional. Then nothing normalised by the clock is quotable from the
# session, and until 2026-09-08 no driver gate said so: the committed H200
# yaml carries only the idle scalar (grade idle-scalar, 1515 MHz) and a session
# run over it would have printed a ruler line and spent every arm. A DONE
# calibrate on this tree writes detail.gemm_clock.median_mhz, so a ruler that
# passed the arm gate and still grades below under-load is a calibrate that
# published without sampling, which is worth three minutes to re-run and not a
# session to spend.
reference_grade_refusal() {
  local card="$1" yaml="$2" grade="$3" mhz="$4" source="$5"
  echo "REFUSED: the ruler for $card carries no clock the roof can be rescaled against."
  echo "  $yaml: reference clock grade '$grade', ${mhz} MHz, usable_for_roof False."
  echo "  source: $source"
  echo "  Every arm below scores LEVEL against this clock and writes a per-row roof"
  echo "  from it. roofline.reference_clock disowns every grade but 'under-load'"
  echo "  (the median sampled while the calibration GEMM ran) for that rescaling, so"
  echo "  against this ruler roof_at_cell_clock_tflops is refused on every row, the"
  echo "  LEVEL verdict is provisional, and NOTHING NORMALISED BY THE CLOCK IS QUOTABLE:"
  echo "  no roof fraction, no LEVEL exclusion, no alpha read off a ladder that"
  echo "  excluded on LEVEL. On the H200 a memory-bound cell boosts to ~1980 MHz"
  echo "  against a ~1515 MHz bf16 GEMM reference, so the fixed-roof fraction is"
  echo "  inflated by up to 31% and the rescaled column is the correction; without"
  echo "  it the session would reproduce the bias the fix round was opened for."
  echo "  Read $LOGS/calibrate.log: a calibrate that lands DONE without writing"
  echo "  detail.gemm_clock.median_mhz did not sample the clock during its GEMM"
  echo "  (nvidia-ml-py missing from $PY_BASE is the usual cause). Fix it, delete"
  echo "  the calibrate row from $LEDGER, and re-run arm 0:"
  echo "      $PY_BASE $REPO/scripts/calibrate_hardware.py --publish"
}

# THE REFUSAL ITSELF, for whichever half of the gate failed, as a function so
# that the WORDS can be planted and read in a test. A refusal that names the
# wrong cause costs the operator the same hour as no refusal at all: the first
# version of this gate had only the yaml half, so a session run with --only
# would have been told its committed ruler was UNDATED when what actually
# happened is that arm 0 was never scheduled.
calibration_refusal() {
  local verdict="$1" card="$2" yaml="$3" state="$4" word="${1#* }"
  local LEDGER_NOTE failed
  LEDGER_NOTE="$(ledger_arm_note calibrate)"
  case "${verdict%% *}" in
   ARM)
    echo "REFUSED: arm 0 did not stand behind a ruler for $card."
    echo "  calibrate is '$word' in $LEDGER; $yaml is $state."
    case "$word" in
      NO_ROW)     if [[ -n "$ONLY" ]]; then
                    echo "  No calibrate row at all: --only $ONLY left arm 0 out, and this"
                    echo "  gate is deliberately NOT scoped to --only. Name calibrate in it,"
                    echo "  first:"
                    echo "      bash scripts/h200_gaps_session.sh --only calibrate,$ONLY"
                  else
                    echo "  No calibrate row at all, and no --only to explain it: arm 0 did"
                    echo "  not run in this session directory. Read $LEDGER."
                  fi ;;
      INVALID)    echo "  It MEASURED and then failed a VALIDITY gate, and it published the"
                  echo "  yaml before scoring one. A clock that could not be established"
                  echo "  makes every number normalised by it unquotable, including the"
                  echo "  ridge every arm below would divide by." ;;
      CLAIM_FAIL) echo "  It measured, every VALIDITY gate passed, and a CLAIM gate did not."
                  echo "  WHICH ONE IS ON ITS OWN PAGE, and this refusal reads it rather"
                  echo "  than naming one: calibrate_hardware.py scores four CLAIM gates"
                  echo "  (no_pattern_exceeds_the_pin_rate, write_rate_is_a_store_rate,"
                  echo "  clock_steady_across_patterns, not_throttled) and until 2026-09-09"
                  echo "  this branch described the first one for all four, and described"
                  echo "  it backwards at that: it fails when a pattern EXCEEDS the"
                  echo "  derived pin rate, not when none reaches it."
                  failed="$(failed_claim_lines "$LOGS/calibrate.log")"
                  if [[ -n "$failed" ]]; then
                    echo "  The lines it printed, verbatim:"
                    printf '%s\n' "$failed" | sed 's/^/    /'
                  else
                    echo "  Its log holds no failing CLAIM line for this driver to quote,"
                    echo "  which is itself a disagreement between the page and the exit"
                    echo "  code: read $LOGS/calibrate.log before believing either."
                  fi
                  echo "  Whichever failed, the yaml it published was written before the"
                  echo "  gate was scored, so it carries the accounting the gate rejected." ;;
      REFUSED)    echo "  It refused before measuring, so the yaml on disk is some earlier"
                  echo "  run's however fresh its stamp reads." ;;
      UNKNOWN)    echo "  The driver would not latch a word for it. The ledger note says why:"
                  echo "      ${LEDGER_NOTE:-(the reason was not recorded on the row)}"
                  echo "  UNKNOWN is five things (moe/bench/exit_codes and second_opinion in"
                  echo "  this file): a DEFECT where the page and the exit code disagree, an"
                  echo "  UNEARNED DONE or INVALID with no RESULT line, an exit 1 from a file"
                  echo "  that has not adopted the table, or a log the second opinion could"
                  echo "  not read. Until 2026-09-08 this branch named only the fourth, which"
                  echo "  calibrate_hardware.py cannot produce (it adopts), and sent the"
                  echo "  operator to the wrong cause at minute 3. Read the log and decide"
                  echo "  by hand; an UNKNOWN row is NOT latched and re-runs on resume." ;;
      *)          echo "  That is not DONE, and DONE is the only state in which this"
                  echo "  instrument stands behind what it wrote." ;;
    esac
    ruler_stakes
    echo "  Read $LOGS/calibrate.log where there is one, fix what it names, and"
    echo "  re-run arm 0:"
    echo "      $PY_BASE $REPO/scripts/calibrate_hardware.py --publish"
    echo "  A CLAIM_FAIL or INVALID row is LATCHED and will not be re-attempted:"
    echo "  delete its row from $LEDGER to force one." ;;
   YAML)
    echo "REFUSED: this session has no calibration of its own for $card."
    echo "  calibrate is DONE in $LEDGER but $yaml is $state."
    case "$word" in
      MISSING)     echo "  Nothing is there. A DONE arm 0 that wrote no tracked file was run"
                   echo "  without --publish: its ruler is in the session path nothing reads." ;;
      UNDATED)     echo "  It is there and carries no provenance.utc, so it cannot say which"
                   echo "  rental measured it. 'Cannot say' is not 'this one'." ;;
      STALE)       echo "  It was measured before this session began, so it is a PREVIOUS"
                   echo "  rental's ridge sitting in this checkout." ;;
      NO_BASELINE) echo "  This driver could not stamp its own start, so it cannot say which"
                   echo "  side of it that file is on. Unset SESSION and take the default." ;;
      *)           echo "  That is not PUBLISHED, and PUBLISHED is the only state in which"
                   echo "  that file is this session's own." ;;
    esac
    ruler_stakes
    echo "  Run arm 0 and let it publish:"
    echo "      $PY_BASE $REPO/scripts/calibrate_hardware.py --publish"
    echo "  then re-run this session; finished arms in $LEDGER are skipped." ;;
   *)
    # A word neither half issues. There is no safe default here: the two states
    # this gate decides between are "this card's ruler" and "another machine's",
    # and guessing either is the defect it exists to prevent.
    echo "REFUSED: calibration_verdict said '$verdict', which this refusal does"
    echo "  not know how to read. Row and yaml state are in $LEDGER and $yaml." ;;
  esac
}

# WHAT A WRONG RULER COSTS, in the words both halves of the calibration gate
# need. Factored so the two refusals cannot drift into saying different things
# about the same consequence.
ruler_stakes() {
  echo "  Every arm below resolves its ridge through roofline.load_measured(),"
  echo "  labels it 'measured on this machine', and would score this card's"
  echo "  roof fractions, LEVEL flags and alphas against another machine's"
  echo "  ceilings. The H200's dense bf16 moved 7.1% between two sessions."
}

# DID THIS --dry-run PRINT A PLAN BEFORE IT REFUSED. rc 0 it did, rc 1 it did
# not, and there is no third answer because the question is about the log.
#
# WHY THE QUESTION EXISTS. Every script this driver runs now exits REFUSED from
# its own --dry-run -- a plan measured nothing and scored no gate, which is what
# the table calls 2 -- so `dry_state` mapping 2 to PLAN_REFUSED gave eleven of
# the sixteen planned arms the same word and DRY mode stopped being able to
# say "this plan is sound" at all. The two states are still different things and
# the log still tells them apart: bm128_depth prints 90 lines of plan and THEN
# says it measured nothing, while bm128_roofline at BLOCK_N=256 prints the
# refusal on line 1 and never plans. So the driver counts the non-blank lines
# printed before the first refusal marker. A script that printed a banner and
# then refused would read as a plan; the fix for that is in the banner, and
# nothing here is a threshold that can be tuned to hide it.
#
# AN ARGPARSE USAGE ERROR IS NOT A PLAN, and it used to read as one. A script
# handed a flag it does not define prints "usage: ..." and "<prog>: error:
# unrecognized arguments: ..." on stderr and exits 2, the driver captures both
# into the log, and every line of that usage text is a non-blank line before a
# refusal marker that never comes: the arm was filed PLANNED. Found on
# 2026-09-09 by giving bm128_depth a flag no version of its script defines; that
# flag is gone from both branches now and no arm below passes one, which
# tests/test_h200_gaps_session.py asserts against every arm's own script rather
# than against a list kept here. The guard stays because the next invented flag
# would be filed PLANNED again. The signature is argparse's own error line,
# which no plan prints.
printed_a_plan() {
  local log="${1:-}"
  [[ -n "$log" && -f "$log" ]] || return 1
  grep -qE '^[[:alnum:]_.-]+: error: ' -- "$log" && return 1
  awk '/REFUSED|NOT A RESULT/ {exit}
       /[^[:space:]]/ {n++}
       END {exit (n > 0 ? 0 : 1)}' "$log"
}

wanted() {
  [[ -z "$ONLY" ]] && return 0
  case ",$ONLY," in (*",$1,"*) return 0 ;; esac
  return 1
}

# WHAT THIS SESSION'S LEDGER HOLDS FOR ONE ARM, last row wins, empty when the
# arm has no row at all. It exists because a FILE landing on disk is not the
# same event as the ARM that wrote it standing behind what it wrote:
# calibrate_hardware.py copies the yaml into the tracked tree BEFORE it scores
# a single gate, and says so in its own --help ("A calibration whose clock
# could not be established is INVALID rather than DONE ... the yaml is still
# written"). So a calibrate that fails VALIDITY clock_established, or the pin
# rate gate, still leaves a fresh-stamped ruler behind it, and a gate that asks
# only WHEN the file was written reads that as this session's own. Asking the
# ledger asks the second question: did the instrument that wrote it pass its
# own gates. Nothing here decides an arm's state; `ledger_state` still does.
ledger_arm_state() {
  awk -F'\t' -v a="$1" '$1 == a { s = $2 } END { print s }' "$LEDGER" 2>/dev/null
}

# THE NOTE AND THE EXIT CODE ON THAT SAME LAST ROW. The note is the one place
# the driver wrote WHY a row is UNKNOWN (second_opinion's five reasons), and a
# refusal that paraphrases one of the five is wrong for the other four the
# moment they exist, which is what calibration_refusal did for five days and
# summarize_arm had already been fixed for. Read, never restated.
ledger_arm_note() {
  awk -F'\t' -v a="$1" '$1 == a { s = $7 } END { print s }' "$LEDGER" 2>/dev/null
}
ledger_arm_rc() {
  awk -F'\t' -v a="$1" '$1 == a { s = $3 } END { print s }' "$LEDGER" 2>/dev/null
}

# THE ONE LINE THE SUMMARY MAY GREP. Anchored at column zero on the prefix
# moe/bench/exit_codes.result_line renders and parse_result_lines reads back.
# A line that merely CONTAINS "PASS", "floor" or "sigma" is prose; the previous
# summary matched eighteen such lines in a REFUSED log and printed an imported
# prior and a pre-registered expectation as if they were this session's output.
result_lines() { grep -E '^RESULT: ' -- "$1" 2>/dev/null; }

# THE CLAIM LINES THAT DID NOT PASS, for a refusal that would otherwise have to
# name a gate. `moe/bench/exit_codes.result_line` renders "RESULT: CLAIM <name>
# <verdict> <detail>", so the gate's own name is on the line and no consumer
# needs a copy of the gate list. The calibration refusal named the pin-rate gate
# for every one of calibrate_hardware.py's four CLAIM gates until 2026-09-09,
# which sent an operator to the wrong number at minute 3 three times in four.
# UNKNOWN is included: classify maps an UNKNOWN CLAIM to CLAIM_FAIL, so an arm
# can wear this word with no FAIL line on its page.
failed_claim_lines() { grep -E '^RESULT: CLAIM [^ ]+ (FAIL|UNKNOWN)' -- "$1" 2>/dev/null; }

# What one arm contributes to the closing summary. Returns 0 when it printed at
# least one RESULT line, 1 when the arm was not scored, so the caller can say
# what the absence means in the mode it is in. $4 is the ledger row's note,
# which for an UNKNOWN row is the reason `second_opinion` would not read a word
# out of the exit code; it is printed rather than paraphrased, because the
# paraphrase this function used to carry ("this arm exited 1 and the file it ran
# has not adopted ...") described one of the five reasons and was wrong for the
# other four the moment they existed.
summarize_arm() {
  local name="$1" log="$2" state="$3" note="${4:-}" hits
  if [[ "$state" == "NOT_PLANNED" ]]; then
    printf '  NOT PLANNED, and not a result. Nothing was run for this arm.\n'
    return 1
  fi
  if [[ ! -f "$log" ]]; then
    printf '  NO LOG at %s: this arm did not run in this session.\n' "$log"
    return 1
  fi
  if [[ "$state" == "REFUSED" || "$state" == "PLAN_REFUSED" ]]; then
    # A COMMAND-LINE REFUSAL IS NOT THE SCRIPT'S OWN REFUSAL, and printing an
    # empty "reason" for it sent the reader to a 90-line log to find one line
    # of argparse on stderr. This branch used to grep only for REFUSED, which a
    # usage error never prints.
    local usage
    usage="$(grep -m1 -E '^[[:alnum:]_.-]+: error: ' -- "$log" 2>/dev/null)"
    if [[ -n "$usage" ]]; then
      printf '  REFUSED BY ITS OWN COMMAND LINE, before the script ran anything:\n'
      printf '    %s\n' "$usage"
      printf '  That is this driver passing a flag the script does not define,\n'
      printf '  not a refusal the script chose. Fix the arm line here or the\n'
      printf '  flag there; nothing about the arm was decided.\n'
      [[ "$state" == "REFUSED" ]] && contract_caveat "$name" REFUSED
      return 1
    fi
    printf '  REFUSED BEFORE MEASURING. Nothing below is a gate:\n'
    grep -m2 'REFUSED' -- "$log" | sed 's/^/    /'
    [[ "$state" == "REFUSED" ]] && contract_caveat "$name" REFUSED
    return 1
  fi
  if [[ "$state" == "UNKNOWN" ]]; then
    printf '  STATE UNKNOWN, NOT LATCHED: the next session attempts this arm again\n'
    printf '  unless you decide otherwise. Why this driver would not read a word\n'
    printf '  out of the exit code:\n'
    printf '    %s\n' "${note:-no note on this row: the reason was not recorded, which is itself a defect}"
    contract_caveat "$name" UNKNOWN
  fi
  if [[ "$state" == "INVALID" ]]; then
    printf '  MEASURED, THEN A VALIDITY GATE FAILED. Nothing from this arm may be\n'
    printf '  quoted, its cells must not be scored, and it is NOT auto-retried:\n'
    printf '  the instrument broke while in use. Re-run it only after the log\n'
    printf '  says in words that the cause was transient.\n'
    contract_caveat "$name" INVALID
  fi
  hits="$(result_lines "$log")"
  if [[ -z "$hits" ]]; then
    return 1
  fi
  printf '%s\n' "$hits"
  return 0
}

# One arm. Runs it, times it, asks git what it did to the tree, and writes one
# ledger row. Never returns non-zero: an arm's verdict belongs in the ledger,
# not in this function's exit status, or `set -o pipefail` and a future `set -e`
# would end the session on the first refusal.
arm() {
  local name="$1"; shift
  if ! wanted "$name"; then
    note "SKIP $name (--only $ONLY)"; return 0
  fi
  local prior
  prior="$(awk -F'\t' -v n="$name" \
    '$1 == n && ($2 == "DONE" || $2 == "CLAIM_FAIL" || $2 == "INVALID") { s = $2 } END { print s }' \
    "$LEDGER" 2>/dev/null)"
  if (( DRY == 0 )) && [[ -n "$prior" ]]; then
    note "SKIP $name (already $prior in $LEDGER; delete its row to force a re-run)"
    return 0
  fi
  local log="$LOGS/$name.log" t0 t1 rc state before after opinion="" adopts=0
  note "-> $name   log $log"
  before="$(dirty_count)"
  t0=$(date -u +%s)
  "$@" > "$log" 2>&1
  rc=$?
  t1=$(date -u +%s)
  after="$(dirty_count)"
  if (( DRY )); then
    state="$(dry_state "$rc" "$log")"
  else
    # THE INTEGER IS THE FIRST OPINION, AND THE PAGE IS THE SECOND. `ledger_state`
    # is still the only thing that turns an exit code into a word; what follows
    # is the driver asking whether the RESULT lines the arm printed agree with
    # that word, and declining to LATCH a word they do not support. The rules
    # are in `second_opinion`, one function, so every branch is plantable. Until
    # 2026-09-03 the only check here was the third pass's "exit 1 from a file
    # that has not adopted the table is UNKNOWN", which covered no measuring
    # arm at all once every one of them adopted, and left a `RESULT: CLAIM C1
    # FAIL` page under exit 0 latched as DONE. The adoption question is asked
    # here, once, and handed over rather than asked twice.
    state="$(ledger_state "$rc")"
    adopts_exit_codes "$(arm_script "$name")" || adopts=$?
    IFS=$'\t' read -r state opinion \
      <<< "$(second_opinion "$rc" "$state" "$(log_verdict "$log")" "$adopts")"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$state" "$rc" "$((t1 - t0))" "$after" "$log" "$opinion" >> "$LEDGER"
  note "   $state (exit $rc) in $((t1 - t0))s; work tree $after dirty file(s)"
  case "$state" in
    BROKEN)  BROKEN_ARMS=$((BROKEN_ARMS + 1))
             note "   BROKEN: this is a PLAN, and it did not survive its own --dry-run."
             note "   $(tail -1 "$log")" ;;
    RETRY)   RETRY_ARMS=$((RETRY_ARMS + 1))
             if [[ -n "$opinion" ]]; then note "   $opinion"
             else note "   exit $rc is not in the table. Read the log before re-running."; fi
             note "   last 5 lines of $log:"
             tail -5 "$log" | sed 's/^/     /' ;;
    UNKNOWN) note "   $opinion"
             contract_caveat "$name" UNKNOWN ;;
    DONE|CLAIM_FAIL)
             note "   $opinion" ;;
    REFUSED|PLAN_REFUSED)
             note "   $(grep -m1 'REFUSED' -- "$log" || tail -1 "$log")"
             [[ "$state" == "REFUSED" ]] && contract_caveat "$name" REFUSED ;;
    INVALID) note "   Do NOT re-run and do NOT quote it: a VALIDITY gate failed after measuring."
             [[ -n "$opinion" ]] && note "   $opinion"
             contract_caveat "$name" INVALID ;;
  esac
  if [[ "$before" != "-" && "$after" != "-" && "$after" -gt "$before" ]]; then
    note "   WARNING: this arm dirtied the work tree ($before -> $after files)."
    note "   Every row it published carries git_dirty=True. Find what it wrote."
  fi
  return 0
}

# An arm that is deliberately not run, named with the reason. Absence in a
# ledger reads as "nothing to report", which is the one thing it never means.
skip_arm() {
  local name="$1" reason="$2"
  wanted "$name" || return 0
  note "NOT PLANNED $name: $reason"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "NOT_PLANNED" "-" "0" "$(dirty_count)" "-" "$reason" >> "$LEDGER"
}

# THE DETECTION LIMIT OF THE WHOLE SESSION, printed before anything is spent.
# Every difference this study has published is one run per condition, so the
# only honest MDE is the known-variance form with sigma imported from the
# replicate floor: scripts/replicate_noise_floor.mde_external_sigma, which is
# 3.96*sigma at n=1. The floor is NOT MEASURED yet -- part (a) has never run on
# a card -- so the sigma is the declared prior in results/published/
# NOISE_FLOOR.json and the line says so in the word ASSUMED. Arm 4 is what
# replaces the assumption with a number.
mde_line() {
  local out rc=0
  out="$("$PY_BASE" - "$REPO" <<'PY' 2>&1
import json
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "scripts"))
import replicate_noise_floor as RNF                                # noqa: E402

path = repo / "results" / "published" / "NOISE_FLOOR.json"
try:
    floor = RNF.noise_floor(path)
    sigma, basis, source = floor.sd, "MEASURED", floor.provenance
except Exception as exc:                                          # noqa: BLE001
    payload = json.loads(path.read_text())
    sigma = float(payload["prior_sd"])
    basis = "ASSUMED"
    source = f'{payload["prior_sd_source"]} [{exc.__class__.__name__}]'
mde = RNF.mde_external_sigma(sigma, 1)
print(f"MDE {mde:.4f} in fitted alpha at the design this study runs "
      f"(one run per condition, known-variance z test, "
      f"level {RNF.TEST_LEVEL}, power {RNF.TEST_POWER}: 3.96*sigma)")
print(f"    sigma {sigma:.4f}, {basis}: {source}")
PY
)" || rc=$?
  if (( rc != 0 )); then
    printf 'MDE UNAVAILABLE: the noise floor could not be read, so this session\n'
    printf '    states no detection limit and every effect below is unscored.\n'
    printf '%s\n' "$out" | tail -3 | sed 's/^/    /'
    return 1
  fi
  printf '%s\n' "$out"
  return 0
}

# Is the attached card pre-Hopper. SM_MAJOR is set once, from a capability that
# matched a number; an unparsed capability leaves it EMPTY and this returns
# false, so an arm is never skipped on a string nobody could read. Written as a
# function because `[[ "${CAPABILITY%%.*}" -lt 9 ]]` evaluates its operands as
# ARITHMETIC: on 2026-09-02 a capability field that had been filled with the
# words "no CUDA" by a delimiter bug made bash look for a variable named CUDA
# and, under `set -u`, end the session there.
pre_hopper() {
  [[ -n "$SM_MAJOR" ]] || return 1
  (( SM_MAJOR < 9 ))
}

# Does git keep this path. ASKED, never asserted from a reading of .gitignore:
# the answer differs for results/, results/published/ and a path outside the
# work tree. rc 128 is a path outside the tree -- the pod default /workspace --
# and calling that "tracked" is how output gets written where nobody collects it.
git_note() {
  local p="$1" rc=0
  git -C "$REPO" check-ignore -q -- "$p" >/dev/null 2>&1 || rc=$?
  case "$rc" in
    0) echo "IGNORED by git -- nothing written here enters the repo (the intended deal for raw output; publish with scripts/publish_results.sh)" ;;
    1) echo "git WILL KEEP this path -- anything written here is committable" ;;
    *) echo "UNVERIFIED: git check-ignore exited $rc. Usually the path is outside this work tree, the normal case for /workspace on a pod. It is NOT 'tracked'." ;;
  esac
}
# WHICH CARD scripts/dram_counter_route.py IS ASKED TO PLAN FOR. That script's
# --card is a HARD DEFAULT of nvidia_a100_sxm4_80gb and is never derived from
# the attached device, so a bare invocation on an H200 prints an A100 cell and
# an A100 ridge. It also takes a CLOSED SET of slugs, so passing a card it does
# not list would exit 2 on argparse and buy nothing: the attached card is used
# only where the file names it, and otherwise the operator is told which
# hypothetical card the page is for. COUNTER_ROUTE_CARD overrides both.
counter_route_card() {
  local want="${COUNTER_ROUTE_CARD:-}"
  if [[ -z "$want" && "${CARD:-nocard}" != nocard ]]; then want="$CARD"; fi
  if [[ -n "$want" ]] && grep -q -- "\"$want\"" "${REPO:-}/scripts/dram_counter_route.py"; then
    echo "$want"; return 0
  fi
  echo nvidia_h200
}

# WHAT EACH ARM COSTS, AND WHERE THE NUMBER CAME FROM. Every figure below is
# what THAT arm's own plan prints when it is invoked the way the lines further
# down invoke it -- not an estimate made here. The old table was wrong by 2.1x
# overall and by 10x on the noise floor, so an operator sizing a rental off it
# under-booked by half a day; the fix is not a better guess, it is asking the
# arm. `arm_basis` names the command and the figure for every row so any of them
# can be re-derived in seconds, and `arm_unpriced` says, in the plan's own
# words, what the figure LEAVES OUT. Three arms are booked ZERO because their
# own plans refuse before any GPU time: a refusal is a result and costs nothing,
# and booking minutes for one hides that the answer is already in.
arm_minutes()  { case "$1" in
  calibrate) echo 3 ;;
  pin_probe-n64-g1) echo 2 ;;   pin_probe-n256-g16) echo 2 ;;
  roofline-n64-g1) echo 1 ;;    roofline-n256-g16) echo 0 ;;
  roofline-n256-g32) echo 0 ;;
  bm128_depth) echo 5 ;;        alias_ablation) echo 14 ;;
  noise_floor) echo 120 ;;
  bn_g16) echo 46 ;;            anchor_measure) echo 5 ;;
  anchor_rescore) echo 0 ;;     occupancy) echo 23 ;;
  mma_switch) echo 7 ;;         ruler) echo 2 ;;         cap_test) echo 5 ;;
  dtype) echo 8 ;;              span_dense) echo 31 ;;   span) echo 0 ;;
  counter_plan) echo 1 ;;
  counter-n32-m64) echo 120 ;;  counter-n128-m64) echo 120 ;;
  counter_contrast) echo 0 ;;
esac; }

# WHERE THAT NUMBER CAME FROM, one line per arm, so no row in the cost table is
# a figure this file invented. Read it as an instruction: run the command and
# the figure is on the page.
arm_basis() { case "$1" in
  calibrate)  echo "calibrate_hardware.py --dry-run prints NO time estimate: a bandwidth ladder, an 8192^3 GEMM per dtype and up to 30 s of settle under load. 3 min is this file's own standing allowance and the one figure here that is not read off a plan." ;;
  pin_probe-n64-g1|pin_probe-n256-g16) echo "moe.bench.cli prints no plan off a GPU box (no framework span registers), so there is no figure to read. 2 min is one profile-cell census under a pin." ;;
  roofline-n64-g1) echo "bm128_roofline.py --dry-run --block-n 64 --group-m 1 --control 256 -> 'estimate 58 s of GPU', 39 cells." ;;
  roofline-n256-g16) echo "bm128_roofline.py --dry-run --block-n 256 --group-m 16 --control 256 --capability 9.0 -> exit 2, 'REFUSED before any GPU time, from the pinned constants alone'. Zero minutes, and the refusal is the arm's finding." ;;
  roofline-n256-g32) echo "the same command at --group-m 32: REFUSED before any GPU time for the same missing BLOCK_M=256 control. Zero minutes." ;;
  bm128_depth) echo "bm128_depth.py --dry-run --model mixtral-8x7b --r-max 2048 -> 'estimate 252 s of GPU', 168 timings (24 treads x 7 reps), for the BM=128 subject against the BM=256 reference. AT --r-max 2048, which is what the pod runs: the default plan is 126 s and a different run id. THE BM=32 PARTNER IS NOT ON THIS COMMAND LINE and never was addable there: it is unconditional in the script (SMALL_TILE_BLOCK_M in BLOCK_SIZES), so it enters the plan by itself. Where the script carries it the same command prints 'estimate 294 s of GPU' and 196 timings, the partner's four extra treads costing 42 s. Both figures book 5 min, and this row quotes the one its own tree prints. Until 2026-09-09 this row named a --partner-block-m flag no version of the script has ever defined, which would have exited 2 on argparse and bought nothing." ;;
  alias_ablation) echo "alias_ablation.py --card 'NVIDIA H200' --models mixtral-8x7b,qwen2-57b-a14b,deepseek-v2-lite,deepseek-v3 --alias-extent block --compute sum --replicates 9 --probe --dot-fallback refuse -> 'WALL 13.7 min ... BOOK THIS ONE', of which 2.0 min is the probe, the figure the page charges outright. Booked 14, above the figure and never at it. THE FALLBACK DOES NOT MOVE THE PAGE: allow and refuse print the same 13.7 because both price the whole ladder; what refuse changes is the cost of a MISS, which is the probe alone. THE SAME COMMAND WITH --run PRINTS THE SAME 13.7 on a box with no GPU, where it refuses at the probe having measured nothing: since 2026-09-03 report_cost charges the probe on the plan page as well, so the dry branch above previews the pod's own booking and the two agree. Until that day the bare plan said 11.6 and this row disclosed the gap; the gap is closed, not disclosed. This row read 13.0 until 2026-09-09 and 13.7 until the sum grid was widened on 2026-09-10; the page now prints 13.7 because the probe compiles ten specialisations instead of six, and 14 is the booking above it. On a GPU box the command WITH --run is the arm itself, so re-derive the row off GPU and without it." ;;
  noise_floor) echo "replicate_noise_floor.py --dry-run --replicates 3 --arms mixtral_g1,mixtral_g16,qwen2_g1,qwen2_g16 -> 'TOTAL: ~120 min of GPU'. Already a WALL figure: that script scales its 3066 s model by the 2.35x wall-over-model factor it measured on the s4 arm." ;;
  bn_g16)     echo "bn_decomposition.py --dry-run --capability 9.0 --group-m 16 --tiles 16,32,64,128 -> 'estimate 2754 s of GPU', 1836 timings (108 treads x 17 reps). 2754 s is 45.9 min, booked 46. THE THIRD SUBJECT TILE IS WHAT MOVED IT, and it is on the command line because that is where the swept set lives: until 2026-09-10 this arm ran the script's default {32, 64, 128} against a 256 reference, and BLOCK_M=128 produced NO alpha at any BLOCK_N (its memory branch came within 15% of its compute branch and was discarded), so six cells over TWO heights were left to fit two parameters. At two heights every candidate extra term correlates +0.72 to +0.98 with the activation column and nothing is identifiable; the same command without 16 prices 2142 s and 1428 timings, which is what this row read until 2026-09-10. BLOCK_M=16 adds 24 treads at 8 per BLOCK_N and is the one added height whose ladder is memory-bound end to end on this card (cap_test swept it to 132 tiles at 9.9% of the roof)." ;;
  anchor_measure) echo "memory_branch_anchor.py --dry-run --measure --model mixtral-8x7b -> 'cells 128 (2 BLOCK_M x 4 G x 16 treads), estimated wall time 4.8 min'. A WALL figure, and the only arm whose plan already charges its compiles." ;;
  anchor_rescore) echo "reads the committed corpus and times nothing. No GPU, seconds." ;;
  occupancy)  echo "occupancy_vs_swizzle.py --dry-run -> 'estimate 1342 s of GPU', 900 timings." ;;
  mma_switch) echo "check_mma_path.sh --dry-run prints its four gates and NO time estimate. 7 min is this file's allowance for two real fused_moe compiles and two PTX dumps." ;;
  ruler)      echo "ruler_rebaseline.py --dry-run -> 'estimated GPU time 105 s (two settles, two GEMMs, two clock samples, two bandwidth passes, one Triton compile)'. That figure NAMES its compile, so nothing is unpriced here." ;;
  cap_test)   echo "tile_cap_test.py --dry-run --capability 9.0 --r-max 2112 -> '81 rows-per-expert x 2 tiles = 162 cells' and 'estimated GPU time 242 s'. AT --r-max 2112, which is what the pod runs: the default takes r_max from depth.rows, which on the H200 band is 688, stops the grid at 672 and leaves two BLOCK_M=256 stacks against V1's three. This row read '96 cells and 143 s' at --r-max 1024 until 2026-09-09; that booking is now REFUSED at plan time by the script's own V4 check, which prints 'the deepest BLOCK_M=16 stack is 66 tiles against the 132 V4 requires' and 'raise --r-max to at least 2112', and prints no cost line to read. 2112 is that printed minimum." ;;
  dtype)      echo "dtype_tile_confound.py --dry-run --card 'NVIDIA H200' -> 'COST 28 cells x 3 arms x 2 dtypes; 32 distinct Triton specialisations; 454 s of timed kernel: 3 repeats x (300 ms warmup + 3 trials x max(200 ms budget, one call))'. 454 s is 7.6 min, booked 8. THE 2026-09-09 RE-SCOPE DOES NOT MOVE THAT FIGURE and the plan page says why: the third arm now transplants BLOCK_SIZE_M and GROUP_SIZE_M only, keeping BN/BK/num_stages feasible for the width it runs at, so all 28 cells stand where the full fp8 transplant was infeasible at 22 of them by the shared-memory arithmetic the plan now prints (SM90_SMEM_LIMIT 232448 against num_stages x (BM*BK + BK*BN) x bytes). Re-read the COST line if that plan changes again; this row is the plan's number, not this file's. It read 315 s until d789b5f charged the warmup as time; the three copies of the old figure in this file were not updated with it, and tests/test_h200_gaps_session.py now pins every KERNEL booking to the figure its plan prints. Without --card it refuses and prints no cost at all." ;;
  span_dense) echo "span_extent_separation.py --dry-run --densify -> '84 cells x 9 arms = 756 timed arms. Estimated KERNEL time 1814 s'." ;;
  span)       echo "span_extent_separation.py --dry-run --no-densify -> 'AND THIS GRID WOULD REFUSE: grid too sparse for C2'. Zero minutes: it stops before it spends one, and that refusal is the extent comparison's honest answer on the published grid." ;;
  counter_plan) echo "dram_counter_route.py --probe returns in seconds and prints no cost line; 1 min is this file's allowance for it. THIS ROW QUOTED 'Budget 15 minutes of GPU time' UNTIL 2026-09-10 and the plan had stopped saying it: the page now reads 'Budget an hour of GPU time and two pod-hours end to end, not the fifteen minutes the one-launch recipe used to promise', because the profiled launch count is warmup + iters x trials rather than one. Neither figure is this arm's, because both price the MEASUREMENT, which is the counter arm below, but a row quoting a sentence its own plan no longer prints is how a booking goes stale without anyone reading it." ;;
  counter_contrast) echo "scripts/dram_counter_route.py --contrast $SESSION/counter_run_n32.json $SESSION/counter_run_n128.json, over the two files the pair above writes. ZERO MINUTES AND NO FIGURE TO READ OFF A PLAN, because it reads two payloads already paid for and times nothing: --contrast is exclusive with --dry-run, so there is no plan page to quote and this row is the command instead. WHY IT IS AN ARM AND NOT A NOTE. Until 2026-09-10 the session paid for both payloads and never took the reading they exist for: grep for --contrast over this driver and the docs returned nothing, and the ratio that decides TRAFFIC from TIME was left to the operator to compute by hand off the printed predictions, which is the improvisation the pair was added to prevent. THE SCORER IS PROVEN OFF GPU on exactly two payloads of the shape --run writes, and its two RESULT lines are VALIDITY X0 and CLAIM XA-all: a traffic world reads as TRAFFIC at a ratio near 1.871 and a time world reads as TIME at 1.000, 87% apart and scored at +/-5% of each rival. IT IS SKIPPED, NEVER REFUSED, WHEN A PAYLOAD IS MISSING: a contrast over one cell is not a contrast, and a half-run pair must not be filed as a failed claim." ;;
  counter-n32-m64|counter-n128-m64) echo "dram_counter_route.py --dry-run --card nvidia_h200 --block-m $(counter_block_m) --block-n 32, and the same at --block-n 128 -> 'COST, of the plan as extended. 5 cells x 6 tile counts x 2 cache modes = 60 profiled invocations of at least 11 fused_experts calls each, about 3300 profiled kernel launches ... At 5 minutes per profiled invocation that is 5.0 GPU-hours for the whole extended plan, against 1.0 for the single cell the plan used to hold, and about half again in pod time ... DROP TO 36 INVOCATIONS (3.0 GPU-hours) by running contrast A alone'. THAT PAGE PRICES FIVE CELLS AND THIS SESSION BOOKS TWO, so neither 5.0 nor 3.0 is this pair's figure and neither may be read as one. WHAT THE PAIR SPENDS, derived from the page's own 5 minutes per profiled invocation rather than transcribed from a sentence: one --run is one cell at one cache mode, which is 6 profiled invocations and 0.5 GPU-hour, so the pair is 12 invocations and 1.0 GPU-hour, exactly the figure the page itself names as '1.0 for the single cell the plan used to hold'. At the page's 'about half again in pod time' that is roughly 45 wall minutes an arm and 90 for the pair. EACH ARM IS BOOKED 120 AND THE PAIR 240, above that figure and never at it, because ncu replay's save and restore of the 2.8 GB weight buffers is the one term in it this repo has never timed. THE COST BLOCK IS BYTE-IDENTICAL AT EVERY BLOCK_N, verified off GPU (md5 c612a3e7a12ebbcd6c65c166316e1cfd at --block-n 32, 64 AND 128), BUT NOT FOR THE REASON THIS ROW USED TO GIVE: the block prices the five-cell extended plan, which does not depend on which single cell you run, so two identical pages no longer license the inference that each cell costs the same, and the 12-invocations-per-arm figure is re-derived above from the per-invocation rate instead of read off that identity. It read md5 fabbeedc38cf2e784316e9e59f6b8f3c until the plan was extended on 2026-09-10. The two plans differ in exactly seven lines: the run id, the pinned line, three corrected-cap rows, the recipe line and the schema's block_n. ONE OF THOSE ROWS DIFFERS IN KIND AND NOT IN VALUE: at BLOCK_N=32 the alpha_a=1 end of the corrected cap reads REFUSED, because alpha_fitted 0.6583 sits below the floor 0.6684 that alpha_b=0 gives, where at BLOCK_N=128 the same row reads 64.1. UNTIL 2026-09-10 IT WAS ONE ARM AT 120 AND THE CONTRAST WAS LEFT TO THE OPERATOR while arm_closes said the arm ran it: that arm line passed no --block-n and no --block-m and took the script's own argparse defaults as they stood that day, BLOCK_N=64 and BLOCK_M=32, which is the single pinned cell the 2026-09-10 analysis named as the defect to fix before running. NO LINE NUMBER IS QUOTED HERE ON PURPOSE: dram_counter_route.py is a separate slice's file and the defaults are its to move, so this row records what the arm RAN and pins the flags on its own line rather than citing a line in a file it does not own. Both flags are on both arm lines now, and the pre-registered discriminator below is at the BLOCK_M they pass." ;;
esac; }

# WHAT THE BOOKED FIGURE DOES NOT INCLUDE, in the arm's own words. Empty for an
# arm whose plan already prints a wall clock. This exists because the two
# largest sweeps in the session say so themselves and the old table read past
# it: span_dense's plan closes with "WALL CLOCK IS NOT THAT NUMBER ... up to 21
# distinct Triton specialisations ... and one weight build per model, the
# largest being deepseek-v3 at 22.5 GB. Budget for those, not for the timings."
# alias_ablation is deliberately NOT listed and its exclusion is deliberately
# EMPTY: its plan prints a KERNEL figure and then a WALL one that charges the
# probe's ten specialisations outright rather than leaving them to the ratio,
# and says BOOK THIS ONE beside it. `report_cost`'s own docstring names this
# file's `arm_unpriced` entry for it as the thing that may then be empty, which
# is the two-call-site defect closed by agreement rather than by a second list.
# No factor is applied to any figure here, because this repo has exactly ONE
# measured wall-over-model datum (127 s logged against 54 s modelled, on one
# small sweep) and multiplying every arm by a small arm's ratio would be an
# invented number wearing a measurement's clothes. The exclusions are printed
# instead, and the total says to book above it rather than at it.
arm_unpriced() { case "$1" in
  roofline-n64-g1|bn_g16|occupancy|cap_test)
              echo "compiles and allocation, in the plan's own words" ;;
  bm128_depth) echo "compiles and allocation, in the plan's own words. The BM=32 scaling partner is NOT an exclusion and is not a flag: it is unconditional in scripts/bm128_depth.py, so its treads are in whatever cell count and estimate that script's own plan page prints, and this row books that figure. 252 s prices the {128, 256} pairing alone; with the partner the same command prices 294 s, four treads and 42 s more, and both are a 5 minute ceiling" ;;
  dtype)      echo "compiles and allocation for 32 distinct Triton specialisations across 2 models" ;;
  span_dense) echo "up to 21 distinct Triton specialisations across 4 models and one weight build per model, the largest deepseek-v3 at 22.5 GB" ;;
  calibrate|mma_switch|pin_probe-n64-g1|pin_probe-n256-g16)
              echo "everything: this row is an allowance, not a figure off a plan" ;;
  *)          echo "" ;;
esac; }

# WHICH CLOCK EACH BOOKED FIGURE IS ON. The total used to add two different
# units and say so nowhere. noise_floor's 120 and anchor_measure's 5 are WALL
# figures -- replicate_noise_floor.py scales its 3066 s model by the 2.35x
# wall-over-model factor it measured, and memory_branch_anchor.py prints
# "estimated wall time" and charges one compile per setting -- while seven arms
# are booked at what their own plans call "the model's own timings, excluding
# compiles and allocation". Adding those into one number and then computing the
# "starts at" column from the sum is how an operator sizes a rental against a
# figure that is not a rental length.
#
# THE UNIT IS READ OFF `arm_unpriced` AND IS NOT A SECOND LIST. A second list
# is this repo's recurring defect: the same fix landing at one of two places
# that name the same set. There is one set here, and it is the exclusions
# already written above -- an empty exclusion means the plan charged everything,
# the allowance sentence means there was no plan to charge, and anything else is
# a kernel-time figure by the arm's own words.
arm_clock() {
  # An arm booked ZERO has no clock to name: its plan refuses before any GPU
  # time, or it reads the committed corpus and times nothing. Calling that WALL
  # would file a refusal as a measured wall figure.
  if [[ "$(arm_minutes "$1")" == 0 ]]; then echo FREE; return; fi
  local unpriced; unpriced="$(arm_unpriced "$1")"
  case "$unpriced" in
    "")            echo WALL ;;
    everything:*)  echo ALLOW ;;
    *)             echo KERNEL ;;
  esac
}

# The ONE measured wall-over-model ratio in this repo, in hundredths because
# this is shell: replicate_noise_floor.py:432 sets WALL_OVER_MODEL = 127/54 from
# its own ARMS.tsv, where the mixtral_g1 arm logged 127 s wall against the cost
# model's 54 s, and applies it to the figure this file books for that arm. It is
# used HERE for one job only: to print, once, what the KERNEL rows would come to
# if they behaved like that one small arm. No per-arm figure is multiplied by
# it. A large sweep amortises its compiles over more timings than a small one
# does, so this is an illustration of the size of the gap and not an estimate of
# any arm. A function and not a bare assignment because this block is lifted
# into a fresh shell by the tests and holds function definitions only.
wall_over_model_pct() { echo 235; }

# The priced total with its KERNEL part put on the wall clock. $1 priced
# minutes, $2 the KERNEL minutes inside them; rounds the scaled part UP, since
# the number exists to be booked above and never at.
bounded_minutes() {
  echo $(( $1 - $2 + ($2 * $(wall_over_model_pct) + 99) / 100 ))
}

# WHAT A RENTAL OF A GIVEN LENGTH BUYS, as two named subsets and the function
# that prices them. They are here rather than in the banner because the banner
# would then hold a second copy of the cost table, and a second copy is this
# repo's recurring defect. `session_bound` walks a list of arm names through the
# SAME `arm_minutes` and `arm_clock` the table above walks, so a re-booked arm
# moves these two lines by itself and nothing has to be remembered.
#
# WHY THESE TWO SETS. The owner's stated payload is bn_g16 and alias_ablation.
# Neither is reachable in a two-hour booking under the full read order, and that
# was true before alias_ablation existed: the noise floor is 120 WALL minutes
# and sits above both of them. So the short sets DROP THE FLOOR rather than
# starting it, because a floor cut short fails replicate_noise_floor's own V2
# and is unquotable, which spends the minutes and buys nothing. calibrate and
# both pin probes are in every set: the calibration gate is not scoped to --only
# and refuses the session without arm 0, and an unhonoured pin makes every
# forced-tile arm below it worthless.
rental_2h_arms() {
  echo "calibrate pin_probe-n64-g1 pin_probe-n256-g16 roofline-n64-g1" \
       "roofline-n256-g16 roofline-n256-g32 alias_ablation bn_g16"
}
rental_3h_arms() {
  echo "$(rental_2h_arms) bm128_depth anchor_measure anchor_rescore" \
       "mma_switch ruler cap_test dtype counter_plan"
}

# The priced total and the bounded total for a set of arms, as two integers on
# one line. Same two functions as the table, same `bounded_minutes`, so a subset
# can never be priced on a different clock from the session it is a subset of.
session_bound() {
  local total=0 kernel=0 n m
  for n in "$@"; do
    m="$(arm_minutes "$n")"
    [[ -n "$m" ]] || { echo "0 0"; return 1; }
    total=$((total + m))
    [[ "$(arm_clock "$n")" == KERNEL ]] && kernel=$((kernel + m))
  done
  echo "$total $(bounded_minutes "$total" "$kernel")"
}

# THE ARMS THE 2026-09-10 SESSION LEFT TO RUN, AND THE COMMAND THAT RUNS THEM.
# REWRITTEN 2026-09-10; the set it replaced was the 2026-09-09 one (calibrate,
# pin_probe-n64-g1, roofline-n64-g1, cap_test, bn_g16, dtype, bm128_depth,
# alias_ablation) and eight of those eight have since been spent. Leaving it
# standing would have re-booked five arms that now hold a RESULT.
#
# WHAT THE 2026-09-10 LEDGER SAYS, since the set is derived from it and not
# chosen: 6 DONE (calibrate, both pin probes, mma_switch, cap_test and
# counter_plan), 5 REFUSED, 5 CLAIM_FAIL and 4 INVALID. A CLAIM_FAIL is a RESULT
# and is never re-run; an INVALID is latched too, and re-running one is only
# worth minutes when the defect behind it has been FIXED. Three of the four
# INVALID rows have not been:
#   roofline-n64-g1  V1 read 0 fresh Triton artefacts at BOTH settings, so the
#                    forced tile is unverified, and V3 still fails on 4 of 39
#                    settling DRIFTs. The fix is a settle-on-clock warmup in the
#                    instrument, not a flag here.
#   alias_ablation   the widened sum grid DID clear the roof this time (headroom
#                    2.120 against a 2.111 bar) and P1 was asked and PASSED, but
#                    `form` failed at R^2 0.5065 on deepseek-v2-lite and
#                    `bracket` at r = 1.392 on mixtral against a 0.9 limit. Both
#                    are gate questions, and re-running the same arm against the
#                    same gates buys the same word for the same 5.4 minutes.
#   noise_floor      V5 failed at an 11.18x cell-to-cell sd ratio against a 3x
#                    bar, and that gate rejects perfectly homogeneous data at
#                    this design most of the time. 120 minutes for the same
#                    word.
# occupancy's INVALID is V9 UNKNOWN (vLLM exposed no fused_moe_kernel Triton
# cache, so the compiled shared memory could not be read); its P1 is a null and
# its P2 is refuted on the cells it did take, and the arm that would advance it
# is a 4x4 GROUP_SIZE_M x num_stages grid this script does not yet run.
#
# SO THE SET IS SIX ARMS AND ONE PRECONDITION. bn_g16 at a third subject tile,
# which is what turns the session's refutation of the activation term into an
# identification; the counter PAIR, the only instrument that reads bytes, and a
# pair because the contrast that decides traffic from time is between two
# BLOCK_N and one cell is not a contrast; counter_plan ahead of both, because a
# BLOCKED probe retires them for the whole session in ten seconds; dtype, whose
# REFUSED cost nothing and was a
# schema collision on a re-used results volume rather than a fact about the
# card; calibrate, because every arm levels against it and a fresh session must
# not read the previous rental's stamp; and pin_probe-n64-g1, because bn_g16 and
# both counter cells pin BLOCK_N and GROUP_SIZE_M and are worth nothing if the
# pin is not honoured.
rerun_arms() {
  echo "calibrate pin_probe-n64-g1 bn_g16 dtype counter_plan counter-n32-m64 counter-n128-m64"
}

# WHAT EACH ONE IS EXPECTED TO REACH, in the ledger's own words, so that the
# next session can be read against a prediction instead of against a hope. A
# CLAIM_FAIL here is a RESULT and is written as one: two of these arms are
# expected to reach it, and an operator who reads exit 1 as a broken arm will
# throw away the finding. Every state below is read off the 2026-09-10 ledger
# and that session's own gate lines, arm by arm, so it is a prediction with a
# basis and not a wish. THE SET IT REPLACED WAS DERIVED THE SAME WAY from the
# 2026-09-09 cells; a rerun table that outlives its session predicts states for
# arms that have since been spent.
rerun_expectation() { case "$1" in
  calibrate)   echo "DONE, 6/6 gates, 29 s. It read DONE on 2026-09-10 and published ridge 155.9 (band 147.9-155.9), 682.1 TFLOP/s bf16 at 1470 MHz and triad 4374.3 GB/s. Re-running it costs half a minute and stamps this session; the card moved 7.1% in dense bf16 between two rentals, so it is not carried." ;;
  pin_probe-n64-g1) echo "DONE. It was DONE in 29 s on 2026-09-10. It is a precondition, not a question, and it gates bn_g16 and both counter cells." ;;
  bn_g16)      echo "CLAIM_FAIL is the LIKELY word and it is a result; DONE is possible and INVALID means the third tile did not qualify. On 2026-09-10 at two subject heights the arm reached V0-V5 PASS, C3 and C5 PASS, and C6/C2/C4 FAIL with C1 UNKNOWN, at alpha_a = -0.8143 +/- 0.0961 against a gate of 0.025 on the spread and [0.10, 0.38] on the value. The 2026-09-10 fit is what the third tile is for: at BLOCK_M in {32, 64} the six cells cannot identify the model, every candidate extra term correlates +0.72 to +0.98 with the activation column, and C2 chi2 is 13.28 over 4 dof. READ V5 AND V2 FIRST: V5 wants >= 3 BN values with an alpha and > 3 cells, and the added BLOCK_M=16 has to qualify a compute reference of its own; if it does, the arm returns NINE cells over three heights and C2 becomes a statement about a term rather than about a two-point degeneracy. 46 min, up from 36 while the swept set was {32, 64, 128}. 16 IS AN ADDED HEIGHT, NOT A REPLACEMENT, and this line said replacement until now: the arm line is --tiles 16,32,64,128, BLOCK_M=128 stays in the swept set, and the plan it prints is 108 treads (8 at BLOCK_M=128 per BLOCK_N among them) for 1836 timings and 2754 s, which is the booking above. arm_basis has always said it the other way, that 16 ADDS 24 treads at 8 per BLOCK_N, so the two descriptions of one swept set disagreed. What 2026-09-10 established is that BLOCK_M=128 yielded no alpha at any BLOCK_N there (its memory branch came within 15% of its compute branch and was discarded), so 16 is the height expected to supply the THIRD memory branch that BLOCK_M=128 did not; it is not booked in its place." ;;
  dtype)       echo "DONE or CLAIM_FAIL on C3/C4, and it is the one arm in this set whose 2026-09-10 word cost nothing: REFUSED at 47 s with ConfoundRefusal, a 70-column timings.csv under a run id whose schema is now 74 columns, on a results volume that outlived the pod. A REFUSED row is re-attempted by every run because refusing is free. ON A FRESH VOLUME IT PLANS AND RUNS; on a re-used one it refuses again with the same line, and the fix it names is --fresh (which discards that file) or a new --run-id (which leaves it alone). It still buys the bf16 native curve, which has never been measured, and a real fp8 native curve. 8 min." ;;
  counter_plan) echo "DONE, about 10 s, and it is in the set to GATE the arm below rather than to be re-asked. On 2026-09-10 it read P1 PASS, route OPEN: ncu 2025.1.1.0 attached with no permission error, cap_eff 0xa80425fb, sys_admin False, the host module flag absent and it attached anyway. BLOCKED here retires BOTH counter arms for this whole session at a cost of ten seconds, which is the reason it runs first." ;;
  counter-n32-m64|counter-n128-m64) echo "DONE or CLAIM_FAIL, and either is the session's headline; NOT_PLANNED if scripts/dram_counter_route.py still defines no --run, which this driver checks before the pod spends an argparse exit 2 on it. What EITHER arm settles on its own: alpha_b as a traffic slope, (dR/dn - a_per_tile)/W, with no fitted level, no delta, no D and no assumed bandwidth: today the same six bn_g16 cells give 0.6087, 0.5930 and 0.5143 depending only on which rate is assumed. What only the PAIR settles, and it is why both are in this set: whether the term the session found in place of the activation re-read is TRAFFIC or TIME. At the BLOCK_M=64 both arms pin, the measured per-M-tile cost is 3.85 GB of weight-set-equivalent at BLOCK_N=32 against 2.06 GB at BLOCK_N=128, so a counter that reads those two figures 1.87x apart says traffic and one that reads the same bytes at both says time. READ THEM TOGETHER OR NOT AT ALL: one arm's bytes-per-M-tile is a number with nothing to be compared against, and a session that runs one of the two has not asked the question. 120 WALL min each, 240 for the pair, off a COST block the plan page prints identically at both BLOCK_N." ;;
  *)           echo "" ;;
esac; }

# The booking, printed. One `say` block rather than a heredoc so that every
# figure comes through `session_bound` and `arm_minutes`, which is what stops
# this from becoming the second cost table.
next_session_booking() {
  local priced bound n exp
  read -r priced bound <<< "$(session_bound $(rerun_arms))"
  printf '\n==== %s ====\n' "THE NEXT SESSION, AND WHY IT IS --new"
  echo "  --resume-latest re-runs no INVALID and no CLAIM_FAIL row: both are"
  echo "  results and both are latched. The 2026-09-10 session landed four of"
  echo "  each, so it resumes into nothing however many defects were fixed in"
  echo "  between. Open a fresh ledger on purpose:"
  echo ""
  echo "      bash scripts/h200_gaps_session.sh --new \\"
  echo "        --only $(rerun_arms | tr ' ' ',')"
  echo ""
  echo "  ~$priced priced / ~$bound bounded minutes, by the same arm_minutes and"
  echo "  arm_clock the cost table above uses. What each arm is expected to"
  echo "  reach, so the ledger can be read against a prediction:"
  for n in $(rerun_arms); do
    exp="$(rerun_expectation "$n")"
    [[ -n "$exp" ]] || continue
    printf '    %-18s %s\n' "$n" "$exp"
  done
  echo ""
  echo ""
  echo "  THREE OF THESE ARE EXPECTED TO EXIT 1 AND ALL THREE ARE RESULTS."
  echo "  CLAIM_FAIL is a RESULT in this table (measured, VALIDITY passed, a"
  echo "  pre-registered CLAIM did not): bn_g16 C1/C2/C6, and the C3 of each"
  echo "  counter arm, are exactly what the card is being rented for, and an"
  echo "  operator who reads exit 1 as a broken arm throws the finding away."
  echo "  An INVALID means the fix for that arm did not land: read its"
  echo "  VALIDITY lines before booking again. THREE ARMS ARE DELIBERATELY"
  echo "  NOT IN THIS SET and their absence is the booking, not an oversight."
  echo "  roofline-n64-g1, alias_ablation and noise_floor each hold an INVALID"
  echo "  whose cause is a gate or an instrument rather than a flag, so"
  echo "  re-running them buys the same word for the same minutes. occupancy"
  echo "  is out for the same reason and its successor is a grid this script"
  echo "  does not yet run."
}

# THE BLOCK_M THE COUNTER PAIR PINS, IN ONE PLACE. The pre-registered
# discriminator is computed at this height and at no other: at BLOCK_M=64 the
# 2026-09-10 ladder slopes give 3.85 GB against 2.06 GB of weight-set-equivalent
# per M-tile, and at the script's own default of 32 the same slopes give 3.53
# and 1.93 at 1.84x. So the arm lines, the skip reason and the closes text all
# read this one function rather than each carrying a 64. A BLOCK_M changed on
# an arm line and not in the paragraph beside it is the defect this pair exists
# to close, in its exact shape.
counter_block_m() { echo 64; }

# THE COUNTER PAIR'S CLOSES TEXT, WRITTEN ONCE AND PARAMETERISED BY BLOCK_N.
# Two arms that differ in one flag are two places for the same paragraph to go
# stale, and a description corrected at one of two call sites is this file's
# standing defect, the one that let a single counter arm be described as
# running a two-cell contrast it did not run. The BLOCK_N is the argument;
# everything else is shared, so a correction lands on both rows or on neither.
counter_closes() {
  local bn="$1" other
  [[ "$bn" == 32 ]] && other=128 || other=32
  echo "THE COUNTER ITSELF, AT BLOCK_N=$bn AND BLOCK_M=$(counter_block_m), one of a PAIR with counter-n$other-m64, and the pair is the session's highest-value booking because it removes a confound nothing else can. WHAT OPEN AND BLOCKED MEAN, and they are the probe arm's two words: OPEN is ncu attaching to this pod with no permission error (2026-09-10: ncu 2025.1.1.0, cap_eff 0xa80425fb, sys_admin False, module flag absent, and it attached anyway), and it is the only state in which this arm is bookable; BLOCKED is the host driver refusing counter collection, which is a FACT ABOUT THE POD and not a broken instrument, and it means this arm buys nothing HERE while the plan stands for the next box. Book counter_plan first in every session and read its P1 RESULT line: OPEN books this arm, BLOCKED does not, and REFUSED (no ncu on PATH) is a third word that says the image is wrong rather than the pod. WHAT IT BUYS THAT NO LADDER CAN. alpha_b = (dR/dn - a_per_tile)/W is a TRAFFIC slope: no fitted level, no delta, no D, and no assumed bandwidth. Today alpha_b is determined only to a factor of two by WHICH RATE IS ASSUMED: the same six bn_g16 cells return 0.6087 at the triad 4374 GB/s, 0.5930 at the session's own measured weight-buffer read rate 4263.5, and 0.5143 at the anchor arm's measured branch rate of about 0.77 of the 4814 pin, and a counter needs none of them. That half is answered by EITHER arm on its own. WHICH OUTCOME DECIDES TRAFFIC VERSUS TIME, and it is why there are TWO arms rather than one and why both pin BLOCK_M=$(counter_block_m). The 2026-09-10 session refuted the three-term model's activation term on the slope alone: that term is the model's only BLOCK_N-dependent one and is strictly proportional to BLOCK_M, so the BLOCK_N dependence must DOUBLE when BLOCK_M doubles, and measured it is 1.115 +/- 0.003, 1.203 +/- 0.007 and 0.923 +/- 0.012 against a required 2.000 (z = -303, -121, -89). What replaces it fits 3.1x better at equal parameter count and goes as 1/BLOCK_N with no BLOCK_M in it: a cost per N-TILE rather than per activation byte. The ladder cannot say whether that cost is TRAFFIC or TIME, and the counter can, in one contrast, but only ACROSS the two arms: one BLOCK_N settles nothing about a BLOCK_N dependence. At BLOCK_M=$(counter_block_m) the measured per-M-tile cost is 1.3676 weight streams at BLOCK_N=32 and 0.7312 at BLOCK_N=128, which is 3.85 GB and 2.06 GB of weight-set-equivalent per M-tile on a 2.81857 GB weight set. IF IT IS TRAFFIC, dram__bytes_read.sum per M-tile lands near those two figures, 1.87x apart, the term belongs inside a byte model, and alpha_b is a function of the N-tile traversal order, which is physically what GROUP_SIZE_M also moves: 24% at a pinned tile, a pinned num_stages and num_warps, and an identical modelled residency of 49152 B per block. IF IT IS TIME, the counter reads the SAME bytes per M-tile at both BLOCK_N within its own few percent, the whole difference sits in gpu__time_duration.sum, and the three-term traffic model cannot contain the term at all: alpha is then measuring a schedule and not a byte count, and the paper says so. Either reading is publishable and neither is available from any ladder this study can run. TWO DEFECTS THE PLAN CARRIED AND THIS PAIR DOES NOT, AND ONE AXIS IT STILL PINS. The registered cell was on nvidia_a100_sxm4_80gb while the route was probed on the H200, so both arms pass --card resolved from the attached device by counter_route_card, and the plan page then reads that card's own ridge (155.93 on the H200 against the 145.81 the A100 default printed). And the plan pinned one schedule, BLOCK_SIZE_M=32 / BLOCK_SIZE_N=64 / GROUP_SIZE_M=16, which is the one axis the session showed the answer depends on: the BLOCK_N half of that pin is what the pair breaks, at the BLOCK_M the discriminator above is registered at, and until 2026-09-10 this line claimed the contrast while the single arm it described passed neither flag and took the script's defaults of 64 and 32. THE AXIS STILL PINNED IS GROUP_SIZE_M, at the script's 16 on both arms, and it is named here rather than left to be read off the plan page: the session measured the per-M-tile cost moving 24% between G=1 and G=16 at byte-identical shared memory and an identical PTX census, so a G=1 counter cell is a THIRD arm and a third two pod-hours, and this session does not book it. VERIFY THE PER-CALL NORMALISATION BEFORE TRUSTING ANY SLOPE: the profiled launch count is warmup + iters x trials fused_experts calls, never one, and the plan's own reduce step divides by the CSV's moe_align_block_size count for exactly that reason."
}

arm_closes() { case "$1" in
  calibrate)  echo "This pod's own ridge and both dtype peaks. Five arms below REFUSE without it, and the H200's dense bf16 moved 7.1% between two sessions, so it is not a constant anything can carry over. It also WRITES a tracked yaml, which is one of the two reasons the dirty-file count is re-asked after every arm." ;;
  pin_probe-n64-g1) echo "The S6a gate ('observed tile_block_m = none') at BLOCK_N=64, GROUP_SIZE_M=1 -- the configuration the control roofline, both bn arms, the anchor and the cap test all pin. Every one of them is worthless if the pin is not honoured." ;;
  pin_probe-n256-g16) echo "The same at BLOCK_N=256, GROUP_SIZE_M=16, the shape vLLM 0.27.1 ships for mixtral at BLOCK_M=128. A pin that reaches the kernel at BLOCK_N=64 is evidence about BLOCK_N=64." ;;
  roofline-n64-g1) echo "THE CONTROL. BLOCK_M=128 at the SWEPT configuration, which production does not ship. It can REFUTE the ceiling (if 128 reaches the roof here, it reaches it everywhere richer) and it CANNOT confirm one for production. Its likely outcome is already predictable from the published G=1 ladders." ;;
  roofline-n256-g16) echo "THE CLAIM'S CONFIGURATION, and NO ARM CAN CONFIRM IT ON sm_90. BLOCK_M=128 at vLLM's own tuned entry for this shape (BLOCK_N=256, GROUP_SIZE_M=16, num_stages 4), which no arm in this study has ever measured. It was scheduled to contest TEMPO's 'the tile term is inactive in decode' in the configuration TEMPO's readers run. No fit, no alpha, no anchor. IT REFUSES AT EVERY WARP AND STAGE COUNT: the BLOCK_M=256 control that cancels the fused layer carries a 256x256 fp32 accumulator, 65536 of 65536 registers per block however the warps are split (bm128_roofline.py --dry-run --block-n 256 --group-m 16 --control 256 --capability 9.0, and the same with --num-warps 16 --num-stages 3, both exit 2), so no pin rescues it and NO BLOCK_SIZE_N confirms the headline on this card. The refusal is the arm's finding: the paper's headline has no confirming arm on the H200, and this driver will not run the subject without its control." ;;
  roofline-n256-g32) echo "The same at GROUP_SIZE_M=32, vLLM's entry at 2048 tokens. Without it the production claim would rest on a single swizzle, and the swizzle is the lever this study has already shown moves alpha by 0.39. Refuses for the same accumulator as the G=16 arm, at every warp and stage count; there is no fix on sm_90 that unblocks either." ;;
  alias_ablation) echo "THE STUDY'S FIRST INFERENTIAL LINK, and the only instrument that tests it. Every alpha here is a slope per extra M-tile RELABELLED as a fraction of a fresh DRAM weight read; every cap, every roof fraction and 'a decode-configured kernel can never reach its compute roof' is that relabelling carried forward, and the relabelling rests on one regression against a byte model with no tile term. This measures the same quantity with no compulsory bytes, no calibrated bandwidth, no ridge and no fitted intercept: one access pattern run twice, one arm's weight loads pointed at an L2-resident column block, alpha = (D(n)/D(1) - 1)/(n-1) with D(1) MEASURED in the same units by the same clock rather than predicted. TWO OUTCOMES, AND BOTH ARE PUBLISHABLE. P1 PASS, the bracket overlapping the refit's 0.529-0.588: the per-tile slope IS DRAM traffic, the mechanism sentence keeps the word, and every cap below keeps its subject. P1 FAIL, the P1 RESULT line saying FAIL in that word and the bracket disjoint from it: the slope is L2-to-shared bandwidth or issue rate or MMA efficiency wearing DRAM's name, alpha_refit is measuring the wrong resource, and the paper's mechanism sentence has to drop the word and say instead which of 0.10 or 0.33 the measured interval did contain. THE THIRD STATE IS NOT AN OUTCOME: headroom or attribution FAILing is INVALID and says the apparatus could not have seen DRAM whatever alpha is, which is exactly what the 2026-09-01 attempt returned and was nearly read as a null result about DRAM. THE FOURTH STATE IS THE LIKELY ONE AND IT IS NOT AN OUTCOME EITHER, and it leaves the ledger with the SAME WORD as the FAIL above. SINCE 2026-09-09 THIS ARM IS BOOKED --dot-fallback refuse, AND THAT IS WHAT THE FOURTH STATE COSTS NOW. On 2026-09-09 it was booked allow, spent 308 s and returned exactly the fourth state: the probe's best sum reading was 5500 GB/s against a 6151 bar (and about 9740 would be needed to satisfy bracket), the run fell to dot mode, P1 read UNKNOWN 'not asked' with alpha >= 0.229, and level, placebo, form and bracket FAILed for reasons that are not the clock rule (one pass of 27 below the band on a folded row, a 28% placebo on a sub-L2 model whose D cannot grow, a true alpha near zero that an R^2 gate cannot pass by construction, and a bracket threshold the probe's own headroom floor was allowed to admit). A dot ladder measures a LOWER BOUND on alpha, cannot ask P1 at all and leaves it UNKNOWN; exit_codes.classify maps an UNKNOWN CLAIM to CLAIM_FAIL; and arm() LATCHES a CLAIM_FAIL, so the allow branch spends thirteen minutes, ends with no answer to P1, and is then SKIPPED by every resume of that session. The bound it buys has now been bought once, so refuse is the booking: if the widened sum grid (4w/4s/64, 4w/6s/128, 2w/4s/128, 2w/5s/64, downward in warps because the ceiling has the cross-lane tl.sum tree's signature) clears the roof, the ladder runs and P1 is asked; if it does not, the arm stops at the probe for about 2.0 min and 3 INVALID rather than for 13 minutes and a latched word bought after the money was spent. ON 2026-09-09 THAT WORD WAS INVALID, NOT CLAIM_FAIL: level, placebo, form and bracket all FAILed, and a validity failure outranks the CLAIM_FAIL an UNKNOWN P1 would otherwise have classified to. Either word latches, and either one costs the same thirteen minutes. READ THE P1 RESULT LINE, NOT THE EXIT CODE: FAIL is the outcome above, and UNKNOWN with a detail opening 'NOT A REFUTATION' refutes no candidate at all, so the 0.10-or-0.33 sentence in the FAIL gloss must not be written from it. A DOT LOWER BOUND IS A DIFFERENT BOOKING AND A DELIBERATE ONE: --compute dot --models mixtral-8x7b,qwen2-57b-a14b,deepseek-v3 --cell-budget-ms 200 --replicates 18 prices 33.4 WALL min on report_cost's own page, and those knobs are what average the 0.1-1 s governor oscillation the 28% placebo came from. It answers P1 with UNKNOWN by construction; book it knowing that. Filing a dot run as 'alpha is not 0.558' when alpha was not asked is the retraction this line exists to prevent. IT NEEDS ARM 0's PUBLISHED CALIBRATION AND DOES NOT REFUSE WITHOUT IT: with no measured yaml for this card both of those gates read UNKNOWN, which is INVALID, so this arm SPENDS its minutes and then may not be quoted. That is a sharper reason for the calibration gate than the five arms that refuse for free." ;;
  bm128_depth) echo "The evaluation's #2: five clean memory-bound treads at 128, monotone. The whole 128 row is currently n=2 across two cards, one on a ladder where time falls as rows rise. IT RUNS AT A DIFFERENT PAIRING SINCE 2026-09-09 and would otherwise be pre-registered INVALID: with the swept set {128, 256} the non-vacuity floor is 2*BM_min/(b*ridge) = 0.838 of the roof, the BM=256 reference measured 0.547 of it on the H200 (365.4 TFLOP/s, one straight line at 1.089 ms/tile for the subject), and no BM=256 ladder in the corpus reaches 0.838 on either card, so the arm refused its own reference before the clock rule was reached. The BM=32 partner in the script's own BLOCK_SIZES puts a small tile in the swept set and the vacuity floor is taken from the CONTROL's measured plateau rather than from the dense GEMM roof. It is unconditional there, not a flag this driver passes: until 2026-09-09 both arm lines booked --partner-block-m 32, which no version of the script defines. What the arm closes is unchanged; what changed is that it can now qualify a reference and decide C1." ;;
  noise_floor) echo "The evaluation's #3: a real between-replicate sd, WRITTEN INTO THE TRACKED TREE. The study has none; every effect so far is scored against an IMPORTED prior, including the MDE this session prints, and until --publish runs that line keeps saying ASSUMED however many replicates were paid for. Also publishes the num_stages control that would have caught the cross-card null. THE ARM SET IS ALL FOUR ARMS AND THAT IS THE DELIBERATE CHOICE, not the default falling through: two models x two swizzles is the SMALLEST set on which this script's own V7 can pass (>= 2 models, or a floor measured only where the swizzle effect is 0.3855 licensing a surface across models where it is 0.0226) and on which either C3 scores a real contrast (a swizzle delta needs G=1 AND G=16 of the SAME model; drop to two arms and C3 reads G=1 against G=1 and measures nothing). The bound is --replicates 3, not a smaller arm set, and it is bought at a stated price: at N=3 the floor ESTIMATE is known to 1.92x by its own table against 1.44x at N=6, so it is published as a floor with that scope attached and a later session extends it rather than re-deriving it." ;;
  bn_g16)     echo "alpha_a as a fitted slope rather than a two-point guess, and the residual that says whether the three-term model is COMPLETE. The only clean lever on the decomposition." ;;
  anchor_measure) echo "The evaluation's weakest link: the memory-branch level, measured at matched reuse rather than extrapolated. Decides whether any numeric alpha is publishable." ;;
  anchor_rescore) echo "Free: every committed report re-scored under the anchor arm 0 just calibrated, so the size of the correction to every published alpha is known. Written under the session directory, never into results/published." ;;
  occupancy)  echo "Whether alpha tracks residency or program order. If residency, the swizzle is a dead lever, the cross-card null is explained, and reuse-distance prediction does not transfer to this regime." ;;
  mma_switch) echo "STUDY item 3's loose end. CLOSES whether the tile alone selects the instruction at fixed tokens." ;;
  ruler)      echo "STUDY item 2's follow-up. Prices the read-vs-triad and clocks-first changes on the committed corpus without adopting them." ;;
  cap_test)   echo "FINDINGS' fourth readout, DEMOTED: BLOCK_M=16 runs multi-tile in 1 of 24 cells on uniform routing, so this tests the formula, not the claim. BOOKED --r-max 2112 SINCE 2026-09-09: at the default the grid held two exactly-full BLOCK_M=256 stacks against V1's three and the arm was unsatisfiable from its own plan page, and the 1024 first booked in its place is itself refused at plan time (V4 wants a 132-tile BLOCK_M=16 stack and 1024 gives 66; the script prints the 2112 minimum). On the 2026-09-09 cells, with the control qualified from its own treads, the counterfactual reads alpha 0.998 raw / 0.994 corrected and a cap of 16.1 Op/B = 0.105 of the ridge, which is a 10x refutation of the retracted 0.10; that is what this arm is now booked to measure rather than replay." ;;
  dtype)      echo "STUDY C2's confound: how much of the 1.15 is the config vLLM resolved differently per dtype. RE-SCOPED 2026-09-09, and the re-scope is the arm: the cross-config arm transplants BLOCK_SIZE_M and GROUP_SIZE_M only, with BLOCK_SIZE_N, BLOCK_SIZE_K and num_stages kept feasible for the width it runs at, and C3/C4 are re-registered against that matched-BLOCK_M arm and dated. The full fp8-config-at-bf16-width transplant is INFEASIBLE on sm_90 and is recorded as such with the arithmetic on the plan page (SM90_SMEM_LIMIT 232448 against num_stages x (BM*BK + BK*BN) x bytes: the fp8 N256/K128 tiles at 3-5 stages ask 294912-409600 B), not run and refused per cell. On 2026-09-09 it WAS run: vLLM 0.27.1's override_config has no try/finally, so the OutOfResources raised inside the context left the fp8 config installed process-wide, all 28 bf16 native cells timed the leaked tile, 13 fp8 native cells timed the previous cell's, and 41 arms were corrupted from one infeasible pairing. The guard, the plan-time refusal and the re-scope are what this arm buys back: the bf16 native curve, which has never been measured, a real fp8 native curve, and C4." ;;
  span_dense) echo "The 0.563 EXTENT-versus-KERNEL split on the DENSE grid, the only grid where C3's mechanism is observable. Runs before the sparse arm because the sparse grid's own kernel world predicts C2 FAIL, and a CLAIM gate failing is a result, not a retry. IT RUNS WHOLE OR NOT AT ALL: --max-minutes 35 used to cap it, which does not refuse -- it breaks out of the cell loop, records the truncation as prose, and lets the gates score a partial grid to a complete grid's exit code. The 31 priced minutes exclude 21 Triton specialisations and four weight builds, so this is the arm most likely to overrun; overrunning honestly is better than a scored fraction of a grid." ;;
  span)       echo "The same on the PUBLISHED grid, booked --no-densify, which is what puts it in a different run id from span_dense: with --densify the default, a bare arm derived the dense arm's id, restored its rows, measured nothing and still landed DONE. IT REFUSES, and its own --dry-run says so in advance: on a powers-of-two grid the padding factor is exactly 1.00 everywhere, so c2_grid_power stops it before it spends a minute. The refusal is the extent comparison's honest answer on that grid, it is free, and it is booked at ZERO rather than at 30 minutes it cannot spend." ;;
  counter_plan) echo "Whether a DRAM counter is reachable here, and it now GATES arms 14 and 15, which are the counter pair. A counter is the only route to alpha_b as a number rather than an interval, and this records which way THIS pod fell. IT IS NOT A FOREGONE BLOCKED ANY MORE and this line said it was until 2026-09-10: the 2026-09-09 and 2026-09-10 RunPod H200s both read OPEN, ncu attaching with no permission error. READ ITS RESULT LINES, AND THE LEDGER WORD IS EARNED: since 2026-09-03 scripts/dram_counter_route.py --probe scores one gate per verdict, prints one RESULT line each, and exits through exit_codes.classify over the same gates, so OPEN and BLOCKED land as the words the table gives them and this driver's second opinion reads the page rather than finding it blank. (Until that day it exited 0 for OPEN and BLOCKED alike with no RESULT line, and before that 3 for everything but OPEN, which filed BLOCKED as INVALID; both halves are fixed.) BLOCKED is the ANSWER, not a broken instrument: a rented pod that cannot reach a DRAM counter is a fact about the pod, recorded so the next session does not spend the minute again." ;;
  counter-n32-m64)  echo "$(counter_closes 32)" ;;
  counter-n128-m64) echo "$(counter_closes 128)" ;;
  counter_contrast) echo "THE READING THE PAIR IS PAID FOR, and until 2026-09-10 nothing took it. Both counter arms write a payload and the discriminator is a RATIO ACROSS them: dR/dn per M-tile at BLOCK_N=32 against BLOCK_N=128 at the same BLOCK_M. TRAFFIC predicts 1.871, TIME predicts 1.000, and the gate scores the measured ratio at +/-5% of each rival, so the two rivals cannot both be within tolerance. EITHER WORD IS A RESULT: TRAFFIC puts the missing term inside a byte model and makes alpha_b a number; TIME says the term is a schedule and no byte model can hold it, and the paper says so. It costs no GPU time, it is scored by the same exit-code table as every arm above, and a pair run half through is SKIPPED here rather than scored, because a claim gate over one cell would file a missing run as a refutation." ;;
esac; }

arm_offgpu_gates() { case "$1" in
  roofline-n64-g1|roofline-n256-g16|roofline-n256-g32)
              echo "scripts/bm128_roofline.py --self-test --fail-on-gate  (three planted worlds, exit 0 required)" ;;
  bm128_depth) echo "scripts/bm128_depth.py --self-test  (three worlds from the law)" ;;
  alias_ablation) echo "scripts/alias_ablation.py --synthetic refit|retracted|tempo|alias-blind  (four planted worlds; 17 RESULT lines each and they SEPARATE by exit code -- refit 0 DONE; retracted and tempo 1 CLAIM_FAIL on P1 with the interval landing on 0.10 and on 0.33; alias-blind 3 INVALID with headroom, attribution, signal and bracket all FAIL, which is the 2026-09-01 apparatus replayed as a planted world. classify_text agrees with the code in all four. THE PLANTED WORLDS RUN IN BOTH DIRECTIONS, which is the point: alias-blind is the world where the gates must NOT report a clean alpha, and the first attempt's 'REFUTED or VOID' is what that world produces.)" ;;
  noise_floor) echo "its plan prints the power table; the floor itself needs replicates" ;;
  bn_g16)     echo "scripts/bn_decomposition.py --self-test --capability 9.0 --group-m 16 --reps 17 --plant-noise 0.008 --tiles 16,32,64,128  (four worlds, four distinct verdicts; exit 0). THE --tiles ARE ON THE GATE LINE BECAUSE THEY ARE ON THE ARM LINE: this file's own standing defect is a gate advertised at one configuration and an arm scheduled at another, which is what the G=1 partner did with a G=16 self-test command. At the swept set this arm runs, S4 reads sd(alpha_a) = 0.0075 against a gate of 0.025 and S5 still fails C2 in the planted MISSING world at chi2 26.22 against a ceiling of 4.0, so the third tile costs the design nothing. AT THE PINNING THIS ARM RUNS AND NO OTHER: the same command at --group-m 1 exits 3 INVALID, which is why there is no longer a G=1 arm for this line to vouch for with a G=16 command." ;;
  anchor_measure|anchor_rescore) echo "scripts/memory_branch_anchor.py --rescore --out-dir <a path outside the tree>  (free, scores every committed report)" ;;
  occupancy)  echo "scripts/occupancy_vs_swizzle.py --self-test, and --audit --fail-on-gate  (A1 FAILs on the corpus by design, exit 1 CLAIM_FAIL). WITHOUT --fail-on-gate the audit prints that FAIL and exits 0, so the advertised check returned the same code whether A1 held or not." ;;
  cap_test)   echo "scripts/tile_cap_test.py --self-test 0.558 and --self-test 0.10" ;;
  span|span_dense) echo "scripts/span_extent_separation.py --self-test kernel|extent|neither --densify --fail-on-world  (15 RESULT lines and exit 0 per world). THE FLAG IS THE CHECK: without it all three worlds exit 2 with ZERO RESULT lines, and the script says so on its own last line -- a gate that examined nothing reporting no failures." ;;
  dtype)      echo "scripts/dtype_tile_confound.py --self-test 2.033|2.400|1.000 --self-test-alpha 0.2 --card 'NVIDIA H200', and --dry-run --card 'NVIDIA H200' for C1 and C2. --card IS THE CHECK OFF A GPU BOX, exactly as --fail-on-world is on span: without it all four commands exit 2 with NoCardToLabel and ZERO RESULT lines, so the operator's pre-rental verification of this arm examined nothing and returned REFUSED, which reads as a broken arm. With it each world prints 10 RESULT lines and C3 SEPARATES them -- 2.033 PASS matched tilt 1.023, 2.400 FAIL 1.208, 1.000 FAIL 0.503. READ THAT SEPARATION, NOT THE EXIT CODE: all three worlds exit 3, because a self-test observes no config and every validity gate but V0 reads UNKNOWN, and classify_text over the log agrees with the 3." ;;
  ruler)      echo "scripts/ruler_rebaseline.py --corpus-only  (hermetic)" ;;
  mma_switch) echo "its four gates need two real compiles; --dry-run registers thresholds only" ;;
  pin_probe-n64-g1|pin_probe-n256-g16) echo "F1 and F2 need a vLLM span, which registers only on the GPU box" ;;
  counter_plan) echo "scripts/dram_counter_route.py --dry-run and --bracket  (the plan and the counter-free bound)" ;;
  counter-n32-m64|counter-n128-m64) echo "scripts/dram_counter_route.py --self-test  (the estimator, off GPU), then --dry-run --card nvidia_h200 --block-m $(counter_block_m) --block-n 32 and the same at --block-n 128, for the cell, the four metrics and the pre-registered per-tile-count predictions. RUN IT AT BOTH BLOCK_N, WHICH IS THE FLAG THE ARM LINE PASSES: a gate advertised at the script's default cell while the arm runs a different one is this file's standing defect, and it is what the single 120-minute counter arm carried until 2026-09-10. THE ESTIMATOR IS THE POINT OF THE SELF-TEST: this arm's alpha_b is (dR/dn - a_per_tile)/W, a traffic slope with no fitted level, no delta and no D in it, so it is not the B/(A+B) that produced every unphysical alpha in the 2026-09-10 session and the self-test is what says so before the card is rented." ;;
  counter_contrast) echo "scripts/dram_counter_route.py --self-test  (the estimator, the ncu parser AND the contrast scorer, off GPU: the self-test carries a THE CONTRAST SCORER section that plants a traffic world, a time world and one it must refuse). There is no --dry-run for this arm: --contrast is exclusive with it, and the predictions it is scored against are already registered on the counter pair's own plan pages, in section 2." ;;
  calibrate)  echo "none: it is a measurement and nothing else" ;;
esac; }
# <<< LIFTABLE

BROKEN_ARMS=0
RETRY_ARMS=0

# --------------------------------------------------------------------------
# The arms, declared once, in the order their results are read. The cost table,
# --list and the closing summary all read this ONE list. The configuration is
# IN THE NAME wherever two arms differ only by configuration, because a ledger
# row that says "roofline" and a report that says BLOCK_N=64 are the same
# defect as a run id without its card.
# --------------------------------------------------------------------------
ARM_NAMES=(calibrate pin_probe-n64-g1 pin_probe-n256-g16
           roofline-n64-g1 roofline-n256-g16 roofline-n256-g32
           bm128_depth alias_ablation noise_floor
           bn_g16 anchor_measure anchor_rescore occupancy
           mma_switch ruler cap_test dtype span_dense span counter_plan
           counter-n32-m64 counter-n128-m64 counter_contrast)

if (( LIST )); then
  say "ARMS, in the order their results are read"
  echo "  Exit codes are moe/bench/exit_codes.py's table for every arm alike:"
  echo "    0 DONE   1 CLAIM_FAIL (a result, never re-run)   2 REFUSED (free)"
  echo "    3 INVALID (measured, unquotable, not auto-retried)   4+ RETRY"
  for n in "${ARM_NAMES[@]}"; do
    printf '\n  %-19s ~%s min %s\n' "$n" "$(arm_minutes "$n")" "$(arm_clock "$n")"
    printf '    %s\n' "$(arm_closes "$n")"
  done
  exit 0
fi

# --------------------------------------------------------------------------
# Preflight. Costs nothing; every refusal here saves a pod hour.
# --------------------------------------------------------------------------
say "PREFLIGHT"
[[ -d "$REPO/scripts" ]] || { echo "REFUSED: no scripts/ under REPO=$REPO"; exit "$RC_REFUSED"; }
[[ -x "$PY_BASE" ]] || { echo "REFUSED: no usable base interpreter (set PY_BASE=)"; exit "$RC_REFUSED"; }
note "repo      $REPO"
note "base py   $PY_BASE"
note "vllm py   $PY_VLLM"

# Every script this driver calls must exist and parse. A missing script would
# otherwise surface as an arm RETRY forty minutes in, with the reason buried.
missing=0
for s in calibrate_hardware bm128_roofline bm128_depth alias_ablation \
         replicate_noise_floor \
         bn_decomposition memory_branch_anchor occupancy_vs_swizzle \
         tile_cap_test dtype_tile_confound span_extent_separation \
         ruler_rebaseline dram_counter_route; do
  if [[ ! -f "$REPO/scripts/$s.py" ]]; then note "MISSING scripts/$s.py"; missing=1
  elif ! "$PY_BASE" -m py_compile "$REPO/scripts/$s.py" 2>/dev/null; then
    note "DOES NOT PARSE scripts/$s.py"; missing=1
  fi
done
[[ -f "$REPO/scripts/check_mma_path.sh" ]] || { note "MISSING scripts/check_mma_path.sh"; missing=1; }
# THE TWO FILES THE LOOP ABOVE NEVER NAMED, and both are load-bearing. Both
# pin_probe arms run `-m moe.bench.cli` (arm_script says so), and the noise_floor
# arm SPAWNS scripts/block_m_crossing_sweep.py twelve times -- once per replicate
# per arm -- so a broken sweep module surfaces as twelve failed replicates two
# hours into the session's largest arm, which is precisely the "arm RETRY forty
# minutes in, with the reason buried" the check above exists to prevent.
for extra in moe/bench/cli.py scripts/block_m_crossing_sweep.py; do
  if [[ ! -f "$REPO/$extra" ]]; then note "MISSING $extra"; missing=1
  elif ! "$PY_BASE" -m py_compile "$REPO/$extra" 2>/dev/null; then
    note "DOES NOT PARSE $extra"; missing=1
  fi
done
if (( missing )); then echo "REFUSED: a script this driver schedules is missing or broken."; exit "$RC_REFUSED"; fi
note "scripts   all 16 present and parse, the thirteen sweep scripts plus"
note "          check_mma_path.sh, moe/bench/cli.py (both pin probes) and"
note "          scripts/block_m_crossing_sweep.py (twelve noise-floor replicates)"

# THE CARD, through moe/bench/provenance so that this file and every report
# agree on the slug character for character. A card this cannot name is the
# literal string "nocard" WITH the reason printed: an unnamed card in a path is
# how one pod's directories were resumed by another pod's session.
CARD="nocard"; CAPABILITY=""; CARD_REASON="not probed"; CARD_OK=0
IFS='|' read -r CARD CAPABILITY CARD_REASON < <("$PY_BASE" - "$REPO" <<'PY' 2>/dev/null
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo))
try:
    from moe.bench.provenance import card_slug, provenance_block
except Exception as exc:                                          # noqa: BLE001
    print(f"nocard||moe.bench.provenance is not importable from {repo}: "
          f"{exc.__class__.__name__}")
    raise SystemExit(0)

prov = provenance_block(repo_root=repo)
if not prov.gpu_name:
    print(f"nocard||{prov.missing.get('gpu_name', 'no reason recorded')}")
    raise SystemExit(0)

capability = ""
try:
    import torch
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    capability = f"{props.major}.{props.minor}"
except Exception as exc:                                          # noqa: BLE001
    capability = ""
    print(f"{card_slug(prov.gpu_name)}||compute capability unreadable: "
          f"{exc.__class__.__name__}")
    raise SystemExit(0)
print(f"{card_slug(prov.gpu_name)}|{capability}|")
PY
)
CARD="${CARD:-nocard}"; CAPABILITY="${CAPABILITY:-}"; CARD_REASON="${CARD_REASON:-}"
[[ "$CARD" != "nocard" ]] && CARD_OK=1
SM_MAJOR=""
[[ "$CAPABILITY" =~ ^([0-9]+)\. ]] && SM_MAJOR="${BASH_REMATCH[1]}"
if (( CARD_OK )); then
  note "card      $CARD${CAPABILITY:+  compute capability $CAPABILITY}"
  [[ -z "$CAPABILITY" ]] && note "          $CARD_REASON"
else
  note "card      nocard -- $CARD_REASON"
  note "          Every path below carries the literal 'nocard', so nothing"
  note "          measured on a real card can resume into this session."
fi

if (( DRY == 0 && CARD_OK == 0 )); then
  echo
  echo "REFUSED: no CUDA device, and this driver measures. Nothing was run."
  echo "  For the review step, off GPU and free:  bash scripts/h200_gaps_session.sh --dry-run"
  exit "$RC_REFUSED"
fi

# CLOCK VISIBILITY, AND IT NOW PROBES THE READER THE INSTRUMENT ACTUALLY USES.
# Every LEVEL and DRIFT verdict comes from `torch.cuda.clock_rate()` in the
# interpreter the arm runs under (moe/bench/timing.py: ClockState.sample), which
# routes through pynvml; nvidia-smi is a different reader in a different
# process, so "nvidia-smi prints a number" was never the question this line
# claimed to answer.
#
# AND IT ANSWERED THE QUESTION IT DID ASK BACKWARDS. On 2026-09-09 this printed
# "nvidia-smi does NOT report clocks on this container" on a pod where
# `nvidia-smi -q -d CLOCK` reports Graphics, SM, Memory and Video clocks. The
# cause is `set -o pipefail` (line 608) over `nvidia-smi ... | grep -q`: grep -q
# exits on the FIRST match, nvidia-smi takes SIGPIPE writing the rest of the
# query, and the pipeline's status is 141, so a match reads as a failure.
# Reproduced off GPU with a 5000-line producer: the branch below took the else
# arm at status 141. No pipeline is used here now, so there is no status to
# invert: the query is captured, matched with bash's own regex, and the two
# readers are reported separately because they can disagree.
if (( CARD_OK )); then
  SMI_CLOCK_Q="$(nvidia-smi -q -d CLOCK 2>/dev/null || true)"
  if [[ "$SMI_CLOCK_Q" =~ Graphics[[:space:]]*:[[:space:]]*[0-9]+[[:space:]]*MHz ]]; then
    note "clocks    nvidia-smi -q -d CLOCK reports a Graphics clock on this container"
  else
    note "clocks    nvidia-smi -q -d CLOCK reports no 'Graphics : N MHz' line here."
    note "          That is what was checked, and it is a container restriction"
    note "          rather than a measurement. Recorded."
  fi
  # The one that decides whether LEVEL and DRIFT are scored at all. A cell whose
  # sampler has no clock source lands `clock_level_ok`/`clock_drift_ok`
  # undetermined, roofline's V3 reads UNKNOWN, its verdict() forces NOT SETTLED
  # on any VALIDITY non-PASS, and pod_session.sh's S6d REFUSES a whole sweep
  # with "the sampler had no clock source". It is worth two seconds here.
  CLOCK_READER="$("$PY_VLLM" - <<'PY' 2>&1 || true
try:
    import torch
    print(f"{int(torch.cuda.clock_rate())} MHz from torch.cuda.clock_rate()")
except Exception as exc:                                          # noqa: BLE001
    print(f"UNREADABLE: {type(exc).__name__}: {exc}")
PY
)"
  case "$CLOCK_READER" in
    *UNREADABLE*)
      note "          THE ARMS' OWN READER IS THE ONE THAT MATTERS AND IT FAILED:"
      note "          $CLOCK_READER"
      note "          torch.cuda.clock_rate() is what moe/bench/timing.py samples"
      note "          under load, through pynvml. Without it LEVEL and DRIFT are"
      note "          undetermined on every cell, roofline V3 reads UNKNOWN and"
      note "          pod_session.sh S6d REFUSES the sweep. Install nvidia-ml-py"
      note "          into $PY_VLLM before spending an arm." ;;
    *) note "          the arms' own reader answers: $CLOCK_READER" ;;
  esac
fi

# The session and results root, with the card IN BOTH NAMES. /workspace/results
# outlives the pod: without the card a second card resumes the first's
# directories and reports the first's timings under the second's heading, which
# is a collision this repo has already published (two arms, two sm_counts, one
# file name). An operator-supplied MOE_RESULTS_DIR is REFUSED rather than
# silently rewritten if it does not name the card.
#
# WHICH SESSION DIRECTORY, through `session_choice`, one function with every
# branch plantable. The ledger, and with it the latch that keeps a finished arm
# from being spent twice, lives in that directory and nowhere else; until
# 2026-09-03 a plain re-invocation opened a fresh one beside it and nothing in
# --help said SESSION= was the way back. SESSION_ROOT is overridable so the
# choice can be exercised against a planted root off GPU.
if [[ -d /workspace ]]; then
  SESSION_ROOT="${SESSION_ROOT:-/workspace/session}"
  SESSION_PREFIX="gaps-$CARD-"
  RESULTS_ROOT=/workspace/results
else
  SESSION_ROOT="${SESSION_ROOT:-$REPO/results/h200_gaps}"
  SESSION_PREFIX="session-$CARD-"
  RESULTS_ROOT="$REPO/results"
fi
IFS=$'\t' read -r SESSION_HOW SESSION_WHAT \
  <<< "$(session_choice "$DRY" "$RESUME_LATEST" "$NEW_SESSION" "${SESSION:-}" "$SESSION_ROOT" "$SESSION_PREFIX")"
case "$SESSION_HOW" in
  NAMED|RESUMED|NEW) SESSION="$SESSION_WHAT" ;;
  LATEST_EXISTS)
    echo "REFUSED: a session for $CARD already exists, with a measuring ledger:"
    echo "  $SESSION_WHAT"
    echo "  A fresh session would open an EMPTY ledger beside that one, re-run every"
    echo "  arm it holds as DONE, and re-run the CLAIM_FAIL and INVALID rows the"
    echo "  latch exists to keep. Nothing was run. Choose, in words:"
    echo "      bash scripts/h200_gaps_session.sh --resume-latest     # skip its finished arms"
    echo "      SESSION=$SESSION_WHAT bash scripts/h200_gaps_session.sh"
    echo "      bash scripts/h200_gaps_session.sh --new               # a fresh ledger, on purpose"
    exit "$RC_REFUSED" ;;
  *)
    echo "REFUSED: ${SESSION_WHAT:-session_choice printed nothing this driver can read}"
    exit "$RC_REFUSED" ;;
esac
if [[ -n "${MOE_RESULTS_DIR:-}" ]]; then
  RESULTS="$MOE_RESULTS_DIR"
  case "$RESULTS" in
    *"$CARD"*) ;;
    *) echo "REFUSED: MOE_RESULTS_DIR=$RESULTS does not contain the card '$CARD'."
       echo "  Two pods share a network volume and one card's run resumes the"
       echo "  other's directories when the card is not in the path. Use"
       echo "  MOE_RESULTS_DIR=$RESULTS/gaps-$CARD or unset it and take the default."
       exit "$RC_REFUSED" ;;
  esac
else
  RESULTS="$RESULTS_ROOT/gaps-$CARD"
fi
export MOE_RESULTS_DIR="$RESULTS"
LOGS="$SESSION/logs"
if (( DRY )); then LEDGER="$SESSION/ARMS-dryrun.tsv"; else LEDGER="$SESSION/ARMS.tsv"; fi
mkdir -p "$LOGS" || { echo "REFUSED: cannot create $LOGS"; exit "$RC_REFUSED"; }
[[ -f "$LEDGER" ]] || printf 'arm\tstate\trc\tseconds\tdirty\tlog\tnote\n' > "$LEDGER"

CALIB_YAML="$REPO/moe/bench/hardware/measured_$CARD.yaml"
# THE INSTANT THIS SESSION BEGAN, as the integer `calibration_state` compares a
# calibration's own timestamp against. Taken from the session directory's name,
# which already carries a UTC stamp, so a session RESUMED into that directory
# still counts the calibration its first pass published instead of refusing an
# hour of finished arms. An operator-supplied SESSION that carries no stamp
# falls back to now, and a resume under such a name refuses until calibrate runs
# again: that is the direction that costs three minutes rather than the session.
SESSION_SINCE="$(utc_stamp "${SESSION##*-}")" || SESSION_SINCE="$(date -u +%Y%m%d%H%M%S)"
DIRTY_AT_START="$(dirty_count)"
note "session   $SESSION"
note "          $(git_note "$SESSION")"
note "results   $RESULTS   (exported as MOE_RESULTS_DIR to every arm; the card is in the name)"
note "          $(git_note "$RESULTS/bm128_roofline")"
note "ledger    $LEDGER"
note "tree      $DIRTY_AT_START dirty file(s) before anything ran, by git status"
note "          --porcelain --untracked-files=all. Re-asked after every arm: two"
note "          arms used to write into the tracked tree while the session ran."
if (( CARD_OK )); then
  note "calib     $CALIB_YAML"
  note "          $(git_note "$CALIB_YAML")   <- THIS ONE is meant to be committed"
fi

# THE COST TABLE, IN BOTH MODES. The TOTAL used to be printed only in the
# `else` of `if (( DRY ))`, which is to say only once the operator had already
# decided to rent: --dry-run listed twenty per-arm minutes and no sum, and the
# sum they would have reached by hand was wrong by nearly four hours. It is
# printed here, in both modes, together with the CUMULATIVE minute each arm
# starts at, because "how long is the session" answers a different question
# than "what is still unstarted when I release the pod". Every row also prints
# where its figure came from and what that figure excludes, so no line of this
# table has to be taken on trust.
#
# AND IT NAMES THE CLOCK ON EVERY ROW, because fixing the figures left the units
# mixed. Seven arms are booked at their plans' "excluding compiles and
# allocation" kernel time and two at a wall clock; the total added both and the
# "starts at" column, the one thing a rental is sized with, was computed from
# that sum. The paragraph under the table said BOOK ABOVE THAT AND NEVER AT IT
# and then gave no number to book above, which left the reader with the mixed
# sum as the only figure on the page. Both numbers are printed now: what the
# plans price, and what the KERNEL part of it comes to at the one wall-over-model
# ratio this repo has measured. Still no per-arm figure is multiplied by
# anything.
say "WHAT THIS COMMITS YOU TO"
note "The min column mixes two clocks and now says which on every row. WALL is a"
note "figure whose own plan charges compiles and allocation. KERNEL is one whose"
note "plan says, in those words, that it does not. ALLOW is this file's own"
note "allowance for an arm whose plan prints no time estimate at all. FREE is an"
note "arm booked zero, whose plan refuses or times nothing, and which therefore"
note "has no clock to be on."
note ""
total=0
kernel_min=0
unpriced_arms=0
for n in "${ARM_NAMES[@]}"; do
  wanted "$n" || continue
  m="$(arm_minutes "$n")"
  clock="$(arm_clock "$n")"
  # The start column is a RANGE, priced start to bounded start, because the two
  # differ by more than two hours by the last arm and the operator sizing a
  # rental needs the second one.
  start_b="$(bounded_minutes "$total" "$kernel_min")"
  if [[ "$start_b" == "$total" ]]; then starts="~$total min"
  else starts="~$total-$start_b min"; fi
  printf '  %-19s ~%3s min %-6s  starts at %-14s  %s\n' \
    "$n" "$m" "$clock" "$starts" "$(arm_closes "$n" | cut -c1-44)"
  printf '  %-19s   from: %s\n' "" "$(arm_basis "$n")"
  excl="$(arm_unpriced "$n")"
  if [[ -n "$excl" ]]; then
    printf '  %-19s   NOT IN THAT FIGURE: %s\n' "" "$excl"
    unpriced_arms=$((unpriced_arms + 1))
  fi
  total=$((total + m))
  if [[ "$clock" == "KERNEL" ]]; then kernel_min=$((kernel_min + m)); fi
done
bound="$(bounded_minutes "$total" "$kernel_min")"
note ""
note "TOTAL ~$total minutes (~$((total / 60))h $((total % 60))m) of what the arms' own plans PRICE,"
note "of which ~$kernel_min are KERNEL minutes: arms whose plans exclude compiles and"
note "allocation in their own words, on rows marked NOT IN THAT FIGURE."
note ""
note "BOOK ABOVE THAT AND NEVER AT IT, AND THIS IS THE NUMBER TO BOOK ABOVE:"
note "~$bound minutes (~$((bound / 60))h $((bound % 60))m), the same table with those $kernel_min KERNEL minutes"
note "multiplied by $(wall_over_model_pct)/100. That factor is the ONE wall-over-model ratio this"
note "repo has measured -- 127 s logged against 54 s modelled for mixtral_g1 on"
note "the s4 arm -- and it is the factor replicate_noise_floor.py already applies"
note "to the $(arm_minutes noise_floor) minutes it books above. IT IS AN ILLUSTRATION OF THE SIZE OF"
note "THE GAP AND NOT AN ESTIMATE OF ANY ARM: one datum, from a small sweep, and a"
note "large sweep amortises its compiles over more timings than a small one does."
note "No figure in the column above was multiplied by it, and $unpriced_arms of the rows above"
note "print a NOT IN THAT FIGURE line naming what their own figure leaves out."
note ""
note "The 'starts at' range is priced start to bounded start, and both are an"
note "EARLIEST start: a rental shorter than an arm's start does not buy that arm,"
note "or anything under it. Nothing in this file has a deadline or a timeout -- an"
note "arm that overruns is not cut short, it pushes everything below it down the"
note "column -- which is why the column is printed at all instead of one number"
note "for the whole session."
note ""
# WHAT FITS, as opposed to what the session costs. The column above answers the
# first question and answered the second nowhere: an operator with a two-hour
# budget could read that bn_g16 starts at minute 146 and had to work out by hand
# which subset to name in --only. Both rows below are priced by `session_bound`
# through the SAME `arm_minutes` and `arm_clock` the table is, so a re-booked
# arm moves them without anything here being edited, and the --only line is
# printed ready to paste rather than described.
read -r RENT2_PRICED RENT2_BOUND <<< "$(session_bound $(rental_2h_arms))"
read -r RENT3_PRICED RENT3_BOUND <<< "$(session_bound $(rental_3h_arms))"
note "WHAT A RENTAL OF A GIVEN LENGTH ACTUALLY REACHES:"
note ""
note "  2 HOURS  ~$RENT2_PRICED priced / ~$RENT2_BOUND bounded min. BOTH PAYLOAD ARMS -- the alias"
note "           ablation and bn_g16 -- with their preconditions and nothing else."
note "             --only $(rental_2h_arms | tr ' ' ',')"
note "  3 HOURS  ~$RENT3_PRICED priced / ~$RENT3_BOUND bounded min. The same plus the depth sweep, the"
note "           anchor pair and the whole cheap tail."
note "             --only $(rental_3h_arms | tr ' ' ',')"
note ""
note "THE 2-HOUR SET NOW BOUNDS ABOVE TWO HOURS, and the honest reading is that"
note "its NAME is the rental and its second figure is not a promise. bn_g16 went"
note "from 36 to 46 priced minutes on 2026-09-10 when a third subject tile was"
note "added to its swept set, and 46 KERNEL minutes bound to 109. The set is"
note "unchanged because what it selects is unchanged: both payload arms and"
note "their preconditions, nothing else. The bound is the ILLUSTRATION"
note "described above and not an estimate of any arm. Read the priced figure as"
note "what the plans charge and the bounded one as the size of what they leave"
note "out; if the card is booked for exactly two hours, expect bn_g16 to be the"
note "arm that is still running when it ends."
note ""
note "NEITHER SET CONTAINS THE NOISE FLOOR, and that is the decision this block"
note "exists to record rather than to hide. It is $(arm_minutes noise_floor) WALL minutes on its own,"
note "it sits above both payload arms in the read order, and a rental that ENTERS"
note "it without finishing it is killed inside it -- which fails that script's own"
note "V2, the planned replicates ran, so even the partial floor is unquotable and"
note "the minutes buy nothing at all. The floor wants a booking of its own, and"
note "until it has one the sigma in the line below stays ASSUMED and every effect"
note "in this study is read against a prior rather than against a measurement."
note ""
note "AND THE ALIAS ARM IS NOT WHAT PUT THEM OUT OF REACH. It is $(arm_minutes alias_ablation) WALL minutes"
note "and it runs ABOVE the floor, so it moved every arm below the floor down by"
note "exactly those minutes and moved nothing above it at all. bn_g16 was already"
note "past a two-hour booking before this arm existed, and the thing that put it"
note "there was the floor. The fix is the --only line, not a shorter arm."
note ""
mde_line | sed 's/^/  /'
note ""
note "  Read every effect this session reports against that limit. The three"
note "  effects already registered in NOISE_FLOOR.json are the swizzle swing"
note "  0.3855 and the footprint spread 0.411, both far above it, and the"
note "  cross-card 0.0117, which is BELOW it and therefore unresolved."
note ""
if (( DRY )); then
  note "--dry-run: each arm runs its OWN --dry-run. Free, off GPU, seconds."
  note "The minutes above are what the POD run would cost, not this one, and"
  note "every one of them is re-derivable from the 'from:' line beside it."
else
  note "The three arms after the pin probes -- roofline-n64-g1 and the two"
  note "BLOCK_N=256 arms that refuse before spending anything -- plus bm128_depth"
  note "are ~$(( $(arm_minutes roofline-n64-g1) + $(arm_minutes roofline-n256-g16) + $(arm_minutes roofline-n256-g32) + $(arm_minutes bm128_depth) )) min and contain the claim. Arms already finished in $LEDGER"
  note "are skipped, so a re-run costs only what is left; a REFUSED arm is"
  note "re-attempted, because refusing costs nothing."
fi

started=$(date -u +%s)

say "SESSION  card=$CARD  $( ((DRY)) && echo '(DRY RUN: plans only)' || echo '(MEASURING)')"

# --------------------------------------------------------------------------
# 0. THIS POD'S OWN CEILINGS, and whether the pin reaches the kernel.
# --------------------------------------------------------------------------
say "0. calibrate THIS card"
if (( DRY )); then
  # IT HAS A --dry-run, AND THIS ARM IS THE ONE THAT MOST NEEDS PREVIEWING. The
  # skip that stood here said "calibrate_hardware.py is a measurement and has no
  # --dry-run", which is false: it prints a nine-line plan -- patterns, buffers,
  # gemm, settle, instrument, card, publish, MDE -- and exits REFUSED, and its
  # own header says so ("`--dry-run` exits REFUSED, because a plan..."). The
  # consequence of the skip was that the ONE arm whose gate can end the whole
  # session at minute 3 was the one arm whose plan an operator never saw before
  # paying for the card.
  arm calibrate "$PY_BASE" "$REPO/scripts/calibrate_hardware.py" --dry-run
else
  # --publish IS THE ARM. Without it this measures this card's ridge into an
  # untracked session path that nothing reads, and every arm below resolves its
  # ridge from whatever measured_*.yaml the LAST rental left in the checkout --
  # 162.81 quoted from another pod while `ridge_source` says "measured on this
  # machine". calibrate_hardware.py's own header says so: "the copy is what
  # makes the ruler visible to roofline.load_measured() and therefore to the
  # sweep". The flag dirties a TRACKED file, which is the cost the A6 fix was
  # avoiding, and that cost is disclosed below rather than paid silently.
  arm calibrate "$PY_BASE" "$REPO/scripts/calibrate_hardware.py" --publish
  # AND THE GATE, in TWO questions, because passing the flag is not the same as
  # the file landing and the file landing is not the same as the file being a
  # ruler.
  #
  # WHEN was it written. `calibration_state` answers MISSING (calibrate refused
  # before it wrote anything), UNDATED (the committed H200 yaml, which carries
  # no provenance block at all) or STALE (the previous rental's). Each of the
  # three means the same thing to every arm below: the ruler is not this card's,
  # so nothing measured against it is this card's either.
  #
  # DID THE ARM THAT WROTE IT PASS ITS OWN GATES. `calibration_state` cannot
  # answer that and a fresh stamp is not the answer: calibrate_hardware.py
  # copies the yaml into the tracked tree BEFORE it scores anything, so a run
  # that lands INVALID on VALIDITY clock_established -- "the samples disagree
  # with the settle plateau; nothing normalised by the clock may be quoted" --
  # or CLAIM_FAIL on the pin rate has still published a fresh-stamped ruler.
  # Asking WHEN alone, this gate printed "measured in THIS session" over a file
  # arm 0 itself refused to stand behind, and every arm below scored every
  # roof fraction, LEVEL flag and alpha against it while `ridge_source` named
  # the attached device. So the ledger row is the second question, and both
  # must answer: DONE, and PUBLISHED.
  #
  # THE GATE IS NOT SCOPED TO --only. `arm calibrate` above returns early when
  # --only names other arms, and this does not, because every arm resolves its
  # ridge through roofline.load_measured() no matter which subset was asked for.
  # A resume into the SAME session directory still passes without re-measuring:
  # the ledger is that directory's, so the first pass's DONE row is still there.
  # The session stops here, where three minutes were spent, instead of at the
  # end, where the whole session was.
  CALIB_STATE="$(calibration_state "$CALIB_YAML" "$SESSION_SINCE")"
  CALIB_ROW="$(ledger_arm_state calibrate)"
  CALIB_VERDICT="$(calibration_verdict "$CALIB_ROW" "$CALIB_STATE")"
  if [[ "$CALIB_VERDICT" == "OK" ]]; then
    note "   ruler     $CALIB_YAML, measured in THIS session by an arm that"
    note "             passed its own gates (calibrate DONE in $LEDGER)."
    note "   It is TRACKED and now dirty, so every row measured below carries"
    note "   git_dirty=True until it is committed. Commit it from another shell"
    note "   before quoting any number that names a commit."
  else
    echo
    calibration_refusal "$CALIB_VERDICT" "$CARD" "$CALIB_YAML" "$CALIB_STATE"
    exit "$RC_REFUSED"
  fi
  # THE THIRD QUESTION: WHAT KIND OF CLOCK does the ruler carry. See
  # reference_grade_refusal for why a fresh, DONE, published ruler can still
  # be one no row below may be normalised against.
  GRADE_LINE="$(reference_grade "$CARD" "$(dirname "$CALIB_YAML")")"
  IFS='|' read -r GRADE_WORD GRADE_MHZ GRADE_USABLE GRADE_SOURCE <<< "$GRADE_LINE"
  if [[ "${GRADE_USABLE:-}" == "True" ]]; then
    note "   reference ${GRADE_MHZ} MHz, grade '${GRADE_WORD}' (${GRADE_SOURCE%%,*})."
    note "             The per-row roof is scored on every row below: read"
    note "             pct_of_roof_at_cell_clock beside pct_of_achieved_tflops, and"
    note "             read clock_level_side on every LEVEL failure. It records"
    note "             the clock that tile held under this card's power cap and"
    note "             excludes nothing on either side: HIGH is a boosted"
    note "             memory-bound cell and LOW is a hungry tile at its own"
    note "             steady state. ONLY DRIFT EXCLUDES A ROW, since 2026-09-09."
  else
    echo
    reference_grade_refusal "$CARD" "$CALIB_YAML" "${GRADE_WORD:-UNREADABLE}" \
      "${GRADE_MHZ:-0}" "${GRADE_SOURCE:-no source line was read}"
    exit "$RC_REFUSED"
  fi
fi

say "0. does MOE_FORCE_TILE reach the kernel AT THE CONFIGURATIONS THE ARMS RUN"
# BLOCK_N=64/GROUP_SIZE_M=1 is block_m_crossing_sweep.FIXED, what the control
# roofline, both bn arms, the anchor and the cap test pin. BLOCK_N=256/
# GROUP_SIZE_M=16 num_stages=4 is vLLM 0.27.1's tuned entry for mixtral at
# BLOCK_M=128 (moe/bench/hardware/vllm_configs/E=8,N=14336,device_name=
# NVIDIA_H200.json, 512 through 4096 tokens), what the production arms pin. The
# probe used to pin BLOCK_N=128, which is neither.
PIN_SWEPT='{"BLOCK_SIZE_M":128,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":64,"GROUP_SIZE_M":1,"num_warps":8,"num_stages":4}'
PIN_PROD='{"BLOCK_SIZE_M":128,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":64,"GROUP_SIZE_M":16,"num_warps":8,"num_stages":4}'
# --env vllm --impl vllm_fused_experts is what makes candidate_impls plan a vLLM
# span at all; without it only torch's CUTLASS spans are planned and the gate
# reads "observed = none" for the wrong reason. Set on this command, never exported.
if (( DRY == 0 )); then
  arm pin_probe-n64-g1 env MOE_FORCE_TILE="$PIN_SWEPT" "$PY_VLLM" -m moe.bench.cli \
      --profile profile-cell --groups baselines --env vllm --impl vllm_fused_experts \
      --out-dir "$SESSION/pin-probe-n64-g1"
elif (( CARD_OK )); then
  arm pin_probe-n64-g1 env MOE_FORCE_TILE="$PIN_SWEPT" "$PY_VLLM" -m moe.bench.cli \
      --profile profile-cell --groups baselines --env vllm --impl vllm_fused_experts \
      --dry-run --out-dir "$SESSION/pin-probe-n64-g1"
else
  skip_arm pin_probe-n64-g1 \
    "no CUDA device: no framework span registers off the GPU box, so the plan would refuse for a reason that is about this laptop and not about the pin."
fi

if pre_hopper; then
  skip_arm pin_probe-n256-g16 \
    "compute capability $CAPABILITY: BLOCK_M=128 x BLOCK_N=256 at 4 stages asks 192 KiB of shared memory against this card's 164. The production configuration is the H200's tuned entry and is not runnable here."
elif (( DRY == 0 )); then
  arm pin_probe-n256-g16 env MOE_FORCE_TILE="$PIN_PROD" "$PY_VLLM" -m moe.bench.cli \
      --profile profile-cell --groups baselines --env vllm --impl vllm_fused_experts \
      --out-dir "$SESSION/pin-probe-n256-g16"
elif (( CARD_OK )); then
  arm pin_probe-n256-g16 env MOE_FORCE_TILE="$PIN_PROD" "$PY_VLLM" -m moe.bench.cli \
      --profile profile-cell --groups baselines --env vllm --impl vllm_fused_experts \
      --dry-run --out-dir "$SESSION/pin-probe-n256-g16"
else
  skip_arm pin_probe-n256-g16 \
    "no CUDA device: no framework span registers off the GPU box, so the plan would refuse for a reason that is about this laptop and not about the pin."
fi

# --------------------------------------------------------------------------
# 1. THE CLAIM, in three configurations. All three run WITH --fail-on-gate, and
#    the sentence that stood here -- "all three run WITHOUT --fail-on-gate: a C1
#    FAIL means BLOCK_M=128 reached the roof, which is one of the two registered
#    outcomes and a result" -- had the conclusion backwards. A failing claim
#    being a RESULT is the argument FOR the flag, not against it: CLAIM_FAIL is
#    the state this ledger latches as finished, and folding it into 0 files it
#    as DONE, "every VALIDITY and CLAIM gate PASSED", over a claim that did not.
#    bm128_roofline no longer has a mode in which a failed gate exits 0 and says
#    the flag is redundant; it is passed anyway, so that a script that grows the
#    downgrade back cannot silently take this arm's CLAIM_FAIL away.
# --------------------------------------------------------------------------
say "1a. CONTROL: BLOCK_M=128 at the SWEPT configuration (BN=64, G=1)"
note "This arm can refute a ceiling and cannot confirm one for production."
if (( DRY )); then
  arm roofline-n64-g1 "$PY_BASE" "$REPO/scripts/bm128_roofline.py" --dry-run \
      --block-n 64 --group-m 1 --control 256
else
  arm roofline-n64-g1 "$PY_VLLM" "$REPO/scripts/bm128_roofline.py" \
      --model mixtral-8x7b --dtype bf16 --control 256 \
      --block-n 64 --group-m 1 --fail-on-gate \
      --r-min 32 --r-max 4096 --reps 3 --plateau-doublings 2
fi

say "1b. THE CLAIM: BLOCK_M=128 at vLLM's own tuned entry (BN=256, G=16)"
# THE CONTROL DOES NOT FIT AT THIS BLOCK_N, AND THAT IS THE ARM'S OWN FINDING.
# bm128_roofline needs a positive control -- the same sweep at a tile with no
# ceiling of its own, which cancels the fused layer's fixed cost, the clocks and
# the occupancy -- and it pins ONE BLOCK_N for the subject and the control alike.
# At BLOCK_N=256 no BLOCK_M=256 control can run on sm_90, by the script's own
# resource model, AT ANY WARP OR STAGE COUNT: the 256x256 fp32 accumulator is
# 65536 registers per block, which is the entire register file, however the
# warps are split. RETRACTED from this note: "256 registers per thread against
# 255 at 8 warps" (RETRACTED), "256 KiB of shared memory against 227 at 16"
# (RETRACTED), and the claim that num_stages 3 or num_warps 16 were escapes
# that merely changed the subject; bm128_roofline now refuses the same command with
# --num-warps 16 --num-stages 3 too, and prints NO TILE ABOVE BLOCK_M=128 CAN
# BE PINNED AT BLOCK_SIZE_N=256 and NONE OF THE FIVE CONFIRMS THE HEADLINE ON
# THIS CARD, AND NO BLOCK_SIZE_N DOES EITHER. So the arm is scheduled with the
# control it needs and REFUSES, free, before any pod time, printing the bill.
# That refusal is the finding: the production configuration has no control at
# its own BLOCK_N on this card, the production claim has a subject and nothing
# to cancel against, and the study cannot confirm its headline at vLLM's
# shipped configuration on sm_90. Running the subject alone would produce
# another uninterpretable "0.5 of the roof", which is the number this whole
# session exists to stop quoting.
if pre_hopper; then
  skip_arm roofline-n256-g16 \
    "compute capability $CAPABILITY: BLOCK_N=256 at 4 stages needs 192 KiB of shared memory against 164. The claim is about the H200 entry and this card cannot run it."
elif (( DRY )); then
  arm roofline-n256-g16 "$PY_BASE" "$REPO/scripts/bm128_roofline.py" --dry-run \
      --block-n 256 --group-m 16 --control 256
else
  arm roofline-n256-g16 "$PY_VLLM" "$REPO/scripts/bm128_roofline.py" \
      --model mixtral-8x7b --dtype bf16 --control 256 \
      --block-n 256 --group-m 16 --fail-on-gate \
      --r-min 32 --r-max 4096 --reps 3 --plateau-doublings 2
fi

say "1c. THE CLAIM at the swizzle vLLM ships at 2048 tokens (BN=256, G=32)"
if pre_hopper; then
  skip_arm roofline-n256-g32 \
    "compute capability $CAPABILITY: same 192 KiB shared-memory refusal as the G=16 arm."
elif (( DRY )); then
  arm roofline-n256-g32 "$PY_BASE" "$REPO/scripts/bm128_roofline.py" --dry-run \
      --block-n 256 --group-m 32 --control 256
else
  arm roofline-n256-g32 "$PY_VLLM" "$REPO/scripts/bm128_roofline.py" \
      --model mixtral-8x7b --dtype bf16 --control 256 \
      --block-n 256 --group-m 32 --fail-on-gate \
      --r-min 32 --r-max 4096 --reps 3 --plateau-doublings 2
fi

say "2. depth at BLOCK_M=128: five clean memory-bound treads, monotone"
# --r-max IS IN THE PLAN AND IN THE RUN ID, so the dry run has to carry it. It
# did not: the preview ran at the default and printed "cells 84 timings / estimate
# 126 s / ...-r1024-...-3a766aa0" while the pod runs "168 timings / 252 s /
# ...-r2048-...-97bf7d2c". A different sweep, a different cost and a different
# run id from the one the operator was shown.
#
# THE BM=32 PARTNER IS NOT A FLAG AND MUST NOT BE BOOKED AS ONE. Until
# 2026-09-09 these two lines passed `--partner-block-m 32`, a flag no version of
# scripts/bm128_depth.py has ever defined: the measuring branch would have exited
# 2 on argparse and the ledger would have filed the one arm this session exists
# to rescue as REFUSED, having spent nothing and measured nothing. The partner is
# UNCONDITIONAL in the script instead (SMALL_TILE_BLOCK_M = 32, carried in
# BLOCK_SIZES beside the subject and the reference), so it is in every plan and
# every run id without either branch naming it.
#
# WHY IT HAD TO BE THERE AT ALL. The 2026-09-09 session spent 292 s and landed
# INVALID on a reference that its own non-vacuity check refused: at block sizes
# {128, 256} the floor is 2*BM_min/(b*ridge) = 0.838 of the roof, the BM=256
# reference ran at 0.547 (365.4 TFLOP/s), and no BM=256 ladder in the corpus
# reaches 0.838 on this card. That is arithmetic, not a measurement: the refusal
# was decidable before the pod was rented and it would repeat on every rerun of
# the {128,256} pairing. A BM=32 partner puts a small tile in the SWEPT set, so
# the floor scales to the smallest tile actually swept, and the vacuity floor is
# derived from the CONTROL's measured plateau rather than from the dense GEMM
# roof.
if (( DRY )); then
  arm bm128_depth "$PY_BASE" "$REPO/scripts/bm128_depth.py" --dry-run \
      --model mixtral-8x7b --r-max 2048
else
  arm bm128_depth "$PY_VLLM" "$REPO/scripts/bm128_depth.py" --model mixtral-8x7b \
      --r-max 2048 --fail-on-gate
fi

say "3. is the per-tile slope DRAM traffic at all, ablated without the byte model"
note "The cheapest arm here that can void the most: 13 WALL minutes against every"
note "alpha, cap and roof fraction the study has published."
# WHY IT RUNS HERE AND NOT AFTER THE FLOOR. Its result changes how bn_g16, the
# anchor, occupancy, cap_test and both span arms are READ -- whether what they
# decompose is a fraction of a weight read or a fraction of something else --
# and the file's ordering rule is that such an arm runs first. The floor changes
# how each of them is SCORED, which is a different question, and the two arms
# are independent in both directions: this one states its own MDE from its own
# pass-to-pass scatter and reads no NOISE_FLOOR.json, and nothing it prints
# changes how a replicate spread is read. So the order between them is a budget
# decision and it is taken in the direction that buys the payload.
#
# EVERY KNOB THAT SETS THE BOOKING OR THE GATES IS NAMED, not defaulted, for the
# same reason NOISE_ARMS is named in the arm below: a later change to a default
# in that script would otherwise silently change what this session buys
# and what it was booked at, with the ledger still reading DONE.
#
#   --models          the four the published alpha was fitted over, so a pooled
#                     number here is comparable with a pooled number there. They
#                     also span 20x in per-expert bytes, which is what makes P2
#                     (alpha tracks per-expert bytes against L2, not a constant)
#                     testable at all. Twenty rungs is where the 13 minutes go.
#   --alias-extent block  THE WHOLE REBUILD. The 2026-09-01 alias zeroed the K
#                     advance too, so every load in the K loop landed on one
#                     16 KiB tile and a handful of L2 slices; that pinning drove
#                     r to between 6.4 and 19.1 where the bracket derivation
#                     needs r < 1, and the run came back VOID for an apparatus
#                     reason that was read as a fact about DRAM. `block` keeps
#                     the K advance real and streams one BLOCK_N x K column
#                     block, L2-resident, over thousands of lines.
#   --compute sum     STUDY.md item 4's own prescription and the UNBIASED
#                     estimator. `dot` is the real GEMM reduction and is biased
#                     LOW, and that script refuses to answer P1 from it.
#   --replicates 9    what the printed MDE is computed at: 0.040 on alpha at the
#                     corpus's median pass spread and 0.095 at its worst,
#                     against the 0.11 that separates 0.10 from 0.33 from 0.558.
#                     Fewer and a FAIL stops being informative.
#   --probe           10% of the arm and it can stop the other 90%. It times a
#                     short pinning grid first and runs the ladder only at a
#                     pinning whose aliased arm clears this card's read roof;
#                     without that headroom no ladder could see DRAM however
#                     alpha falls, which is the first attempt's whole story.
#   --dot-fallback refuse  what the probe may do when NO sum-mode pinning clears
#                     the roof. CHANGED FROM `allow` ON 2026-09-09, after the
#                     first H200 session bought the `allow` branch and got what
#                     it buys: the probe's best sum reading was 5500 GB/s
#                     against a 6151 bar, the run fell to dot mode, and the
#                     ladder came back with P1 UNKNOWN ("not asked", alpha >=
#                     0.229) plus level, placebo, form and bracket FAILs, 308 s
#                     spent, latched INVALID. `allow` takes the fastest dot-mode
#                     pinning, which measures a LOWER BOUND on alpha, cannot ask
#                     P1 at all and exits 1 CLAIM_FAIL with "the claim was not
#                     asked" on the page; `refuse` stops at the probe for 3
#                     INVALID at the cost of the probe alone, about 2.0 min.
#                     Both states are LATCHED by this ledger, so the choice is
#                     between spending thirteen minutes for a bound that was
#                     already taken and spending the probe to find out whether
#                     the WIDENED sum grid clears the roof. The widening is the
#                     other half of this booking and it is in the script: the
#                     sum half of PROBE_PINNINGS now goes DOWNWARD in warps
#                     (4w/4s/64, 4w/6s/128, 2w/4s/128, 2w/5s/64), because the
#                     0.61-of-roof ceiling has the signature of the cross-lane
#                     tl.sum tree and fewer warps is the direction that shortens
#                     it. If the probe clears, the ladder is the arm as booked.
#                     THE DOT LOWER BOUND IS STILL AVAILABLE AND IS NOT THIS
#                     ARM: `alias_ablation.py --run --compute dot --models
#                     mixtral-8x7b,qwen2-57b-a14b,deepseek-v3 --cell-budget-ms
#                     200 --replicates 18` costs 33.4 WALL min by report_cost's
#                     own page, and those three knobs are what average out the
#                     0.1-1 s governor oscillation that produced the 28%
#                     placebo FAIL on deepseek-v2-lite. Book it deliberately,
#                     in a session of its own, knowing P1 stays UNKNOWN.
#
# IT DEPENDS ON ARM 0 AND DOES NOT REFUSE WITHOUT IT, which is a stronger reason
# for the calibration gate than the five arms that do. Its `headroom` and
# `attribution` gates divide by this card's measured read roof, and with no
# tracked calibration they read UNKNOWN, which is INVALID: the arm MEASURES for
# thirteen minutes and then may not be quoted. Its own plan says so in one line
# ("NO CALIBRATION for this card ... run scripts/calibrate_hardware.py --publish
# first"), and the gate after arm 0 is what stops the session before it gets
# here.
#
# THE DRY BRANCH IS BARE ON PURPOSE, AND SINCE 2026-09-03 IT NO LONGER
# UNDER-BOOKS ITSELF. That
# script does not take a --dry-run flag at all: a bare invocation IS its plan
# and --run is what makes it measure. Until 2026-09-03 `report_cost` charged
# the probe's ten specialisations only under --run, so the plan the branch
# below prints said WALL 11.6 where the pod spent WALL 13.0. The script now
# charges the probe on the plan page as well, so the bare plan IS the pod's
# figure. Passing --run in a --dry-run was never the fix: off this laptop it
# refuses at the probe, but a --dry-run session on the pod would then MEASURE,
# and this file's rule is that a plan is free in every sense. The booking is
# the 13, `arm_basis` names the command that prints it and says the two agree,
# and there is no difference left for the operator to read.
#
# AND OFF A GPU BOX IT NEEDS A CARD TO NAME, exactly as dtype does: without one
# it labels the run "no-card-nothing-measured", declines to price the arm at all
# ("NOT PRICED. There is no committed calibration for this card") and says its
# two card gates will read UNKNOWN. The hypothetical is LABELLED, in the run id
# and in every row: if the card that gets rented is not one, the plan was for a
# different machine.
ALIAS_MODELS=mixtral-8x7b,qwen2-57b-a14b,deepseek-v2-lite,deepseek-v3
ALIAS_PLAN_CARD="${ALIAS_PLAN_CARD:-NVIDIA H200}"
if (( DRY )) && (( CARD_OK )); then
  arm alias_ablation "$PY_BASE" "$REPO/scripts/alias_ablation.py" \
      --models "$ALIAS_MODELS" --alias-extent block --compute sum \
      --replicates 9 --probe --dot-fallback refuse
elif (( DRY )); then
  note "   no CUDA device: planning alias_ablation for the HYPOTHETICAL card"
  note "   '$ALIAS_PLAN_CARD'. Its ceilings, run id and cost are that card's,"
  note "   not this laptop's and not necessarily the one you rent."
  arm alias_ablation "$PY_BASE" "$REPO/scripts/alias_ablation.py" \
      --card "$ALIAS_PLAN_CARD" \
      --models "$ALIAS_MODELS" --alias-extent block --compute sum \
      --replicates 9 --probe --dot-fallback refuse
else
  # No --card on the pod: the device names itself and a flag would override it.
  arm alias_ablation "$PY_VLLM" "$REPO/scripts/alias_ablation.py" --run \
      --models "$ALIAS_MODELS" --alias-extent block --compute sum \
      --replicates 9 --probe --dot-fallback refuse
fi

say "4. the noise floor, without which nothing above can be scored"
# THE THREE FLAGS ARE THE ARM. Run bare, this was booked 25 minutes against its
# own plan's 240, ran to no deadline in a session with no timeout anywhere, and
# published nothing.
#
#   --replicates 3   is the BOUND, and the owner's decision is that the floor
#                    stays in the main run rather than moving to a session of
#                    its own. 3 not 6: the plan's own table prices 6 at 240 min
#                    and 3 at 120, and the price of the bound is stated rather
#                    than hidden -- the floor ESTIMATE is known to 1.92x at N=3
#                    against 1.44x at N=6 (8 df against 20). Power is not what
#                    is being bought here and never was: the plan says N=6 "is
#                    NOT set by power -- power alone is satisfied at N = 2 for
#                    both big effects", it is set by how well the floor itself
#                    is known. A floor known to 1.92x is a floor; the 240-minute
#                    version is a better one that this session cannot afford
#                    without spending everything below it.
#   --arms           all four, NAMED rather than defaulted. Naming them is what
#                    stops a later change to that script's default from silently
#                    changing what this session buys and what it was booked at.
#                    Four is the SMALLEST set that scores: V7 needs >= 2 models
#                    (a floor measured only where the swizzle effect is 0.3855
#                    does not license a surface across models where it is
#                    0.0226, and V7 is a VALIDITY gate, so a single-model set
#                    lands the whole 120 minutes on 3 INVALID unless
#                    --single-model-floor is also given), and each C3 needs G=1
#                    AND G=16 of the SAME model -- pick two arms one per model
#                    and the plan still prints two C3 rows, now reading G=1
#                    against G=1, a contrast across nothing.
#   --publish        IS THE ARM, exactly as it is for calibrate. NOISE_FLOOR.json
#                    is written under that flag and no other path reaches
#                    write_published, so without it the most expensive
#                    measurement in the session leaves the sigma in every future
#                    MDE line still reading "0.0228, ASSUMED" -- the number this
#                    arm exists to replace.
#
# --publish IS NOT ON THE DRY-RUN LINE, and that is not an oversight. Verified
# on this laptop: `replicate_noise_floor.py --dry-run --replicates 3 --arms ...
# --publish` WRITES results/published/NOISE_FLOOR.json, which git tracks. A plan
# that dirties the tracked tree is the anchor-rescore defect this file already
# carries a paragraph about, and a dry run must be free in every sense.
NOISE_ARMS=mixtral_g1,mixtral_g16,qwen2_g1,qwen2_g16
if (( DRY )); then
  arm noise_floor "$PY_BASE" "$REPO/scripts/replicate_noise_floor.py" --dry-run \
      --replicates 3 --arms "$NOISE_ARMS"
else
  arm noise_floor "$PY_VLLM" "$REPO/scripts/replicate_noise_floor.py" \
      --replicates 3 --arms "$NOISE_ARMS" --publish
fi

# --------------------------------------------------------------------------
# 5-7. WHAT alpha IS MADE OF.
# --------------------------------------------------------------------------
say "5. alpha_a and alpha_b separated, and whether the three-term model is complete"
# ONE ARM, AT GROUP_SIZE_M=16, BECAUSE THAT IS THE ONLY PINNING AT WHICH THIS
# INSTRUMENT IS VALID. There used to be a G=1 partner, booked 11 minutes against
# its own plan's 36 and advertised off GPU with the G=16 self-test command --
# which is the exact substitution that script's own S5 note records the audit
# as having caught, put back one line later. Run the pinning the arm actually
# ran through the arm's own gate and it does not pass:
#
#   bn_decomposition.py --self-test --capability 9.0 --group-m 1 --reps 17 \
#       --plant-noise 0.008        -> exit 3 INVALID
#     S4 FAIL  sd(alpha_a) = 0.1759 at GROUP_SIZE_M=1 against a gate of 0.025
#     S5 FAIL  the planted MISSING world PASSES C2 at chi2 1.78 against 4.0,
#              so C2 cannot fail there however the card behaves
#     S2 FAIL  which is the same fact read from the other side
#
# and the arm's own plan says the rest in words: "AT THIS PINNING C1 WILL READ
# UNKNOWN however the run goes". The arm had been reframed as an alpha_b + C3
# arm rather than an alpha_a arm, which would be a fair answer if anything
# bounded alpha_b's spread there -- S1 checks alpha_b's LEVEL on one planted
# draw and nothing checks its SPREAD, so the reframed arm has no registered
# detection limit for the one number it is for.
#
# RE-PINNING WAS THE OTHER OPTION AND THERE IS NOWHERE TO RE-PIN IT TO. The same
# self-test at --group-m 2, 4, 8, 32 and 64 also exits 3, with sd(alpha_a)
# 0.1759, 0.1759, 0.4864, 0.1759 and 0.0635 against the same 0.025, and --reps
# does not buy it either: 65 reps reaches 0.1021 and 129 reaches 0.0902, four
# times the gate for eight times the minutes. GROUP_SIZE_M=16 is the one setting
# that passes, and it is the arm below. So the minutes go back to the session
# rather than to a second arm at a pinning its own design gate refuses.
# A THIRD SUBJECT TILE SINCE 2026-09-10, AND IT IS THE ARM'S WHOLE POINT NOW.
# On 2026-09-10 this arm ran the script's default swept set {32, 64, 128} with a
# 256 reference and landed CLAIM_FAIL with SIX usable cells over TWO heights:
# BLOCK_M=128 produced no alpha at any BLOCK_N, because its memory branch came
# within 15% of its own compute branch and every cell was discarded. Two heights
# is not enough grid to identify the model. Every candidate extra term then
# correlates +0.72 to +0.98 with the activation column, four different terms all
# drop the chi2 below the bar, and the fit leaves alpha_a at -3.12, -4.00, +0.71
# and -0.14 depending on which one is added, on a quantity that must lie in
# [0, 1]. The BN-scaling test refuted the activation term outright at 89 to 303
# sigma, and refuting is all a two-height grid can do: it cannot say what
# replaces it.
#
# BLOCK_M=16 IS THE ADDED HEIGHT AND NOT BLOCK_M=128, and the reason is on the
# card. 128's branch is the one that would not separate; 16's is memory-bound
# end to end here, which cap_test measured on the same day by sweeping it to 132
# M-tiles at 9.9% of the roof. So --tiles 16,32,64,128 buys three heights that
# yield a memory branch where the old set bought two, and it breaks the BM/BN
# versus 1/BN collinearity the whole decomposition currently founders on.
# It costs 24 treads: the plan goes from 1428 timings and 2142 s to 1836 and
# 2754, which is the 46 minutes this arm is booked at.
#
# SCORE IT ON THE SLOPE, NOT ON B/(A+B). The estimator, not the data, produced
# every unphysical alpha in the 2026-09-10 session. Of the 23 ladders that
# session fitted, FIVE have a negative intercept and FIVE return B/(A+B) above
# 1.0 on a quantity that cannot exceed it, and this arm's own alpha_a of
# -0.8143 has a sign that lies entirely inside the reference fixed cost's
# leave-one-tread-out error. `alpha_upper = B/(A+B-D)` exceeds 1 exactly when D > A, which held in
# four of these six cells and in no others: arithmetic, not physics.
if (( DRY )); then
  arm bn_g16 "$PY_BASE" "$REPO/scripts/bn_decomposition.py" --dry-run \
      --capability "${CAPABILITY:-9.0}" --group-m 16 --reps 17 \
      --tiles 16,32,64,128
else
  # On sm_80 it needs --num-stages 3: BM=256 x BN=128 at 4 stages asks 192 KiB
  # against 164. The script refuses and names the fix; we pass it up front.
  # The empty-array expansion is guarded: under set -u, bash before 4.4 reads
  # "${STAGES[@]}" of an empty array as an unbound variable and ends the
  # session at this arm (verified on /bin/bash 3.2). Ubuntu's bash 5 does not,
  # which is why it was never caught on a pod; the guard costs nothing.
  STAGES=(); [[ "$SM_MAJOR" == "8" ]] && STAGES=(--num-stages 3)
  arm bn_g16 "$PY_VLLM" "$REPO/scripts/bn_decomposition.py" --group-m 16 --reps 17 \
      --tiles 16,32,64,128 --fail-on-gate ${STAGES[@]+"${STAGES[@]}"}
fi

say "6. the memory-branch anchor, measured rather than extrapolated"
# THE DRY RUN HAS TO CARRY --measure, or it previews the wrong mode of the arm.
# Without it the script defaults to rescore -- "mode : --rescore (no GPU, no
# measurement, seconds)" followed by 26 committed reports -- so the arm this
# file calls the evaluation's weakest link had its plan verified by nothing, and
# its dry-run row duplicated the anchor_rescore arm skipped two lines below.
# With it the plan is a different page: "cells 128 (2 BLOCK_M x 4 G x 16
# treads), estimated wall time 4.8 min".
if (( DRY )); then
  arm anchor_measure "$PY_BASE" "$REPO/scripts/memory_branch_anchor.py" --dry-run \
      --measure --model mixtral-8x7b
else
  arm anchor_measure "$PY_VLLM" "$REPO/scripts/memory_branch_anchor.py" --measure --model mixtral-8x7b
fi
# Free, and it SCORES: --rescore reads the committed corpus and returns a gate
# verdict, so it is a measurement of the corpus and not a plan. It is therefore
# not run in a --dry-run session, where a CLAIM_FAIL would be recorded as a
# broken plan. --out-dir keeps it out of results/published, whose two tracked
# files it rewrote on every previous invocation, --dry-run included.
if (( DRY )); then
  skip_arm anchor_rescore \
    "--rescore scores the committed corpus and returns a gate verdict, which is a result and not a plan. Run it directly: scripts/memory_branch_anchor.py --rescore --out-dir <path outside the tree>."
else
  arm anchor_rescore "$PY_BASE" "$REPO/scripts/memory_branch_anchor.py" --rescore \
      --out-dir "$SESSION/anchor-rescore"
fi

say "7. residency or program order: does the standard predictor transfer"
if (( DRY )); then
  arm occupancy "$PY_BASE" "$REPO/scripts/occupancy_vs_swizzle.py" --dry-run
else
  # No stage override: --stages is the residency LADDER, and the script prunes
  # rungs the attached card cannot hold (s=5 on sm_80) from its own capability.
  #
  # --fail-on-gate IS LOAD-BEARING HERE and was missing. occupancy_vs_swizzle's
  # own exit_for returns DONE for a CLAIM_FAIL without it, and P2 is EXPECTED to
  # fail -- that FAIL is this arm's finding. Proven off GPU, live:
  # `occupancy_vs_swizzle.py --audit` prints "RESULT: CLAIM
  # A1_reuse_distance_on_the_corpus FAIL", then "reported as exit 0 without
  # --fail-on-gate", and returns 0, while exit_codes.classify_text over that log
  # returns 1. With the flag both are 1. So this arm's registered outcome was
  # unreachable: it could land DONE, REFUSED or INVALID and nothing else.
  arm occupancy "$PY_VLLM" "$REPO/scripts/occupancy_vs_swizzle.py" --run --fail-on-gate
fi

# --------------------------------------------------------------------------
# 8-12. THE DOCS' OWN BACKLOG, in its previous order, after the claim.
# --------------------------------------------------------------------------
say "8. the ISA switch: is the instruction selected by the tile alone"
if pre_hopper; then
  skip_arm mma_switch "compute capability $CAPABILITY reaches no warpgroup MMA at any tile."
elif (( DRY )); then
  arm mma_switch env MOE_PYTHON="$PY_VLLM" bash "$REPO/scripts/check_mma_path.sh" --dry-run \
      --model deepseek-v3 --tokens 256 \
      --block-m 16,64 --block-n 64 --block-k 64 --group-m 1 --warps 4 --stages 3
else
  # ONE call, two tiles: --block-m takes a list, and everything else is held
  # fixed, so the only thing that moves between the two PTX dumps is BLOCK_M.
  # THE INTERPRETER IS HANDED ACROSS. check_mma_path.sh chooses its python from
  # MOE_PYTHON or /workspace/venvs/vllm/bin/python, and this driver resolves
  # PY_VLLM with fallbacks and passed it to every vLLM arm but this one, so on
  # a pod where PY_VLLM is overridden the MMA arm silently compiled under a
  # different interpreter from the pin probes it is read beside, and off a GPU
  # it refused on a missing /workspace path instead of on CUDA like every
  # other arm. `env` is a plain command, so `arm` still captures its exit.
  arm mma_switch env MOE_PYTHON="$PY_VLLM" bash "$REPO/scripts/check_mma_path.sh" \
      --model deepseek-v3 --tokens 256 \
      --block-m 16,64 --block-n 64 --block-k 64 --group-m 1 --warps 4 --stages 3 \
      --out "$SESSION/ptx/mma-switch"
fi

say "9. the ruler: price a ridge change without making one"
if (( DRY )); then
  arm ruler "$PY_BASE" "$REPO/scripts/ruler_rebaseline.py" --dry-run
else
  arm ruler "$PY_BASE" "$REPO/scripts/ruler_rebaseline.py" --fail-on-gate
fi

say "10. BLOCK_M=16 cap test -- the FORMULA, not the production claim"
# --r-max IS THE ARM, and leaving it to the default is what made the 2026-09-09
# run INVALID before a cell ran. `r_max = args.r_max or depth.rows`
# (tile_cap_test.py) took the H200 band's depth requirement, 688 rows; 688 % 32
# = 16, so the grid stopped at 672 and held TWO exactly-full BLOCK_M=256 stacks
# against V1's requirement of three aligned treads per tile. V1 was
# unsatisfiable from the plan page, which printed "BM=256:2" and continued;
# compute_reference then skipped the two-tread control, fell through to the
# BM=16 subject and refused it on vacuity, and V2 failed on an identity nothing
# printed.
#
# BOOKED 2112, NOT 1024. Between 2026-09-09 and now these two lines read
# --r-max 1024, which is the value tile_cap_test.py's own plan-time refusal
# rejects: `--dry-run --capability 9.0 --r-max 1024` prints "REFUSED: the grid
# cannot satisfy its own validity gates", "the deepest BLOCK_M=16 stack is 66
# tiles against the 132 V4 requires" and "raise --r-max to at least 2112", and
# prints no cost line at all. Booking it would have bought the same nothing the
# default bought, one gate further along. 2112 is the script's own printed
# minimum: the plan is 81 rows-per-expert x 2 tiles = 162 cells, stacks
# BM=16:69 and BM=256:8, the deepest BLOCK_M=16 stack exactly the 132 V4
# requires, and 242 s of kernel. THE PLAN CARRIES IT TOO: --r-max is in the
# grid, in the cost and in the run id, so a dry branch without it previews a
# different sweep, the same defect this file fixed on bm128_depth.
if (( DRY )); then
  arm cap_test "$PY_BASE" "$REPO/scripts/tile_cap_test.py" --dry-run \
      --capability "${CAPABILITY:-9.0}" --r-max 2112
else
  arm cap_test "$PY_VLLM" "$REPO/scripts/tile_cap_test.py" --fail-on-gate --r-max 2112
fi

say "11. is the 1.15 fp8/bf16 crossing the FORMAT or the CONFIG"
# THIS ARM'S PLAN NEEDS A CARD TO NAME, AND OFF A GPU BOX IT HAS TO BE GIVEN
# ONE. Run bare, `dtype_tile_confound.py --dry-run` refuses with "NoCardToLabel:
# no CUDA device, no --card and no --gpu-name, so there is no card to derive
# vLLM's config lookup for", the ledger records PLAN_REFUSED, and that reads as
# "this arm is broken" rather than "this laptop has no GPU" -- the distinction
# this file draws explicitly for the pin probes at the top of the session and
# did not draw here.
# The refusal names its own fix, and supplying it plans the arm and prints the
# cost the operator was never shown: "COST 28 cells x 3 arms x 2 dtypes; 36
# distinct Triton specialisations; 454 s of timed kernel".
#
# The hypothetical is LABELLED, not smuggled: the run id, the config lookup and
# every row of a plan made this way say NVIDIA H200, and if the card that gets
# rented is not one, the plan was for a different machine. On a GPU box no
# --card is passed at all, because the device names itself and a flag would
# override it.
DTYPE_PLAN_CARD="${DTYPE_PLAN_CARD:-NVIDIA H200}"
if (( DRY )) && (( CARD_OK )); then
  arm dtype "$PY_BASE" "$REPO/scripts/dtype_tile_confound.py" --dry-run
elif (( DRY )); then
  note "   no CUDA device: planning dtype for the HYPOTHETICAL card"
  note "   '$DTYPE_PLAN_CARD'. Its config lookup, run id and cost are that"
  note "   card's, not this laptop's and not necessarily the one you rent."
  arm dtype "$PY_BASE" "$REPO/scripts/dtype_tile_confound.py" --dry-run \
      --card "$DTYPE_PLAN_CARD"
else
  arm dtype "$PY_VLLM" "$REPO/scripts/dtype_tile_confound.py" --fail-on-claim
fi

say "12. is the 0.563 separation the span EXTENT or the KERNEL"
note "Dense grid first: it is the only grid on which C3's mechanism is observable."
# THE SPARSE ARM IS BOOKED --no-densify, AND THAT IS THE WHOLE FIX. --densify
# became the default on 2026-09-02 (argparse BooleanOptionalAction), so the bare
# arm densified too: both arms derived the SAME run id, the second restored
# every row the first had written, timed nothing, and landed DONE in the ledger
# with 30 minutes booked against it. The id is a hash of the plan and `densify`
# is one of its knobs, so naming the flag is what separates them -- verified off
# GPU, `--densify` derives ...-densifytrue-...-e95805af and `--no-densify`
# ...-densifyfalse-...-5f329a66. --max-minutes could not have separated them:
# it prices a run rather than defining one and is deliberately not in the key.
# The sparse arm now REFUSES before measuring, by that script's own c2_grid_power
# gate ("grid too sparse for C2"), which is free and is the honest answer for
# the grid V5's argument prefers.
# --max-minutes IS GONE FROM BOTH ARMS, and that is a refusal to buy a partial
# grid at a complete grid's price. The flag does not refuse and it does not
# refuse LATER either: at span_extent_separation.py:4602 it breaks out of the
# cell loop, sets `stopped = "stopped after N minutes with K of M cells done"`,
# and that string is printed as prose and stored under a "partial" key. No gate
# reads it. V4 asks only that "real work happened -- cells/samples/compiles > 0",
# which a third of a grid satisfies, so a truncated run and a whole one exit
# with the same code and the reduction scores whichever cells happened to
# finish. span_dense was booked 20 min, capped at 35, against a plan that prices
# 1814 s of KERNEL and then says the wall clock is not that number; the likely
# outcome of the 0.563 extent-versus-kernel arm was therefore a partial dense
# grid scored as if complete.
#
# WHERE THE REAL FIX LIVES, and it is not here. Making a truncation REFUSE
# belongs in span_extent_separation.py, at the break and in the gate list,
# because only that file knows WHICH cells are missing and whether its
# crossing estimator can stand without them -- a grid truncated after the small
# token counts is a different loss from one truncated after the large. A driver
# cannot tell those apart from an exit code. What a driver CAN decide is
# whether to arm the truncation at all, and arming it is what converted a
# complete-grid arm into a partial one wearing a complete one's exit code. So
# the arms run whole, the honest time is booked, and if the pod is released
# mid-arm the ledger shows an arm that never finished rather than one that
# finished on a third of the evidence.
if (( DRY )); then
  arm span_dense "$PY_BASE" "$REPO/scripts/span_extent_separation.py" --dry-run --densify
  arm span       "$PY_BASE" "$REPO/scripts/span_extent_separation.py" --dry-run --no-densify
else
  arm span_dense "$PY_VLLM" "$REPO/scripts/span_extent_separation.py" --densify
  arm span       "$PY_VLLM" "$REPO/scripts/span_extent_separation.py" --no-densify
fi

say "13. is a DRAM counter route open on this box"
# THE CARD IS NAMED AT EVERY CALL SITE, AND THAT IS THE POINT OF THIS BLOCK.
# scripts/dram_counter_route.py takes --card from a HARD DEFAULT of
# nvidia_a100_sxm4_80gb, not from the attached device, so on 2026-09-10 the
# probe reported OPEN on an H200 and the plan printed underneath it named an
# A100 and an A100 ridge of 145.81. The verdict and the cell were about
# different machines on one page. `counter_route_card` resolves the flag ONCE,
# from the attached card wherever the script lists it, and all SIX invocations
# read that one value: this arm's two branches, and one per branch in each of
# the two counter arms below, which `counter_arm` writes once for both.
# A card resolved at one of several call sites is this repository's recurring
# defect and it is what produced that page. --probe is the one invocation
# without it, because it takes no card and prints no cell.
COUNTER_PLAN_CARD="$(counter_route_card)"
if (( DRY )); then
  arm counter_plan "$PY_BASE" "$REPO/scripts/dram_counter_route.py" --dry-run \
      --card "$COUNTER_PLAN_CARD"
else
  arm counter_plan "$PY_BASE" "$REPO/scripts/dram_counter_route.py" --probe --out "$SESSION/counter_route.json"
  # The plan is printed regardless: if the probe said OPEN it is the command to
  # type next; if BLOCKED it is what to run on the box where it is not.
  "$PY_BASE" "$REPO/scripts/dram_counter_route.py" --dry-run --card "$COUNTER_PLAN_CARD" \
      >> "$LOGS/counter_plan.log" 2>&1 || true
fi

say "14 and 15. the DRAM counter, run at two BLOCK_N"
# THE ARMS THE PROBE ABOVE EXISTS TO BOOK, and there are TWO of them because
# one BLOCK_N settles nothing about a BLOCK_N dependence. They are the only
# instrument in this study that reads bytes rather than time, and everything
# they settle is settled with no fitted level, no delta, no D and no assumed
# bandwidth. `arm_closes counter-n32-m64` says what OPEN and BLOCKED mean and
# which outcome decides traffic versus time; this block is the mechanics.
#
# THE CONTRAST IS ON THE ARM LINES, AND UNTIL 2026-09-10 IT WAS NOT. This block
# booked ONE arm named `counter` that passed no --block-n, no --block-m and no
# --group-m, so it ran the script's own argparse defaults as they stood on
# 2026-09-10 (BLOCK_N=64, BLOCK_M=32, GROUP_SIZE_M=16), the
# single pinned cell the 2026-09-10 analysis named as the second of two defects
# to fix before running, while `arm_closes` said in the same file that the arm
# ran "at TWO BLOCK_N at fixed BLOCK_M rather than at one" and listed that pin
# among the defects the arm did not carry. `arm_basis` said the other thing in
# the same breath, that a second BLOCK_N is a second booking the operator makes.
# A description left standing after the behaviour it describes is this
# repository's recurring defect, and here it was two descriptions of one arm
# disagreeing inside one file. The pair below runs it: --block-m 64 on both,
# --block-n 32 and 128, which is the BLOCK_M the pre-registered discriminator
# (1.3676 against 0.7312 weight streams per M-tile, 3.85 GB against 2.06 GB,
# 1.87x) is registered at. The two plans carry different run ids
# (...-n32-m64-... and ...-n128-m64-...) so the two arms land in different run
# directories, and their COST blocks are byte-identical, so the pair is priced
# at 2 x 120 = 240 WALL minutes off the same page.
#
# GROUP_SIZE_M IS STILL PINNED AT 16 ON BOTH, and that is a booking decision
# rather than an oversight: a G=1 cell is a third arm and a third two pod-hours.
# `arm_closes` names it as the axis the pair still pins.
#
# ONE GATE, TWO ARMS. The card, the probe gate and the runner check are written
# ONCE, in `counter_arm`, and both rows go through it. Writing the gate twice is
# how a fix lands on one call site: this file's own history has both depth arm
# lines carrying a --partner-block-m that no version of scripts/bm128_depth.py
# ever defined, and it has the single counter arm this block replaces.
#
# --card IS PASSED AND IS NOT COSMETIC. That script's --card is a hard default
# of nvidia_a100_sxm4_80gb and is never derived from the attached device, so on
# 2026-09-10 the page printed an A100 ridge of 145.81 beside a verdict measured
# on a card whose own ridge is 155.93. COUNTER_PLAN_CARD is resolved once by
# `counter_route_card`, above arm 13, and every invocation reads that one value.
#
# THE RUNNER IS CHECKED BEFORE THE POD SPENDS AN ARGPARSE EXIT 2 ON IT. This
# driver's own standing defect is a flag written on an arm line that the script
# does not define: on 2026-09-09 both depth arm lines carried --partner-block-m,
# which no version of scripts/bm128_depth.py has ever defined, and the arm the
# session existed to rescue would have exited 2 having measured nothing. So the
# measuring branch asks the file whether it defines --run, exactly as
# `adopts_exit_codes` asks whether it imports the exit-code table, and NAMES the
# absence rather than discovering it on the card.
#
# AND THE PROBE GATES BOTH, WHICH IS WHY counter_plan RUNS ONE LINE ABOVE. The
# probe costs ten seconds and answers exactly the question that licenses these
# arms' four pod-hours, so a BLOCKED or REFUSED verdict has to stop the spend
# rather than be read off the page afterwards. The gate reads THIS session's
# ledger row, not the log and not a remembered result from another rental: a
# counter route is a property of the pod, and the pod is what changes between
# sessions. It is skipped rather than refused, because nothing is broken when
# a host declines counter collection.
counter_arm() {
  local name="$1" bn="$2"
  if (( DRY )); then
    arm "$name" "$PY_BASE" "$REPO/scripts/dram_counter_route.py" --dry-run \
        --card "$COUNTER_PLAN_CARD" --block-m "$(counter_block_m)" --block-n "$bn"
  elif [[ "$(ledger_arm_state counter_plan)" != DONE ]]; then
    skip_arm "$name" \
      "counter_plan is $(ledger_arm_state counter_plan), not DONE, so no counter route was confirmed OPEN on this box in this session. DONE is the OPEN verdict; CLAIM_FAIL is BLOCKED, which is a fact about the pod and not a broken instrument; REFUSED is no ncu on PATH, which is the image. This arm is 120 wall minutes and it can measure nothing without the route, so it is skipped rather than spent. Read $LOGS/counter_plan.log for which of the three it was."
  elif ! grep -q -- '"--run"' "$REPO/scripts/dram_counter_route.py"; then
    skip_arm "$name" \
      "scripts/dram_counter_route.py defines no --run: it plans, probes, brackets and analyses, and the ncu loop is still the shell recipe its own plan page prints. Run that recipe by hand from the plan in $LOGS/counter_plan.log, at --block-m $(counter_block_m) and BOTH --block-n 32 and 128 or the contrast is not run, and reduce it with --analyse; this arm books the runner and refuses to invent it."
  else
    arm "$name" "$PY_VLLM" "$REPO/scripts/dram_counter_route.py" --run \
        --card "$COUNTER_PLAN_CARD" --block-m "$(counter_block_m)" --block-n "$bn" \
        --out "$SESSION/counter_run_n$bn.json"
  fi
}
counter_arm counter-n32-m64  32
counter_arm counter-n128-m64 128

# THE READING THE PAIR IS PAID FOR. Added 2026-09-10, when a build audit ran
# `grep -rn -- --contrast scripts/h200_gaps_session.sh docs/*.md` and got
# nothing: the session booked four pod-hours to write two payloads and then left
# the ratio between them to the operator, off the printed predictions, by hand.
# That is the same shape as an arm that measures and files no verdict, and it is
# the one arithmetic step the whole pair exists to make possible.
#
# ZERO MINUTES, OFF GPU, AND AFTER BOTH. It reads two files already on disk and
# times nothing, so it is booked FREE and placed last in the read order. Off a
# GPU box neither payload exists, and `--contrast` is mutually exclusive with
# `--dry-run` in the runner, so there is no plan page for this arm to print and
# it is NOT_PLANNED there with the reason rather than PLAN_REFUSED: nothing
# about the plan was refused, the inputs are simply not written yet.
#
# A HALF-RUN PAIR IS SKIPPED, NOT SCORED. `--contrast` over one payload REFUSES
# by design, but a REFUSED row here would read as this session having asked the
# question and been told no. It was never asked. The missing file is named
# instead, which is the same discipline `skip_arm` exists for everywhere else.
counter_contrast_arm() {
  local name="$1" lo hi
  local missing=()
  lo="$SESSION/counter_run_n32.json"
  hi="$SESSION/counter_run_n128.json"
  if (( DRY )); then
    skip_arm "$name" \
      "off a GPU box neither payload exists: $lo and $hi are written by counter-n32-m64 and counter-n128-m64 on the card, and --contrast is exclusive with --dry-run so there is no plan page to print. The predictions this arm is scored against ARE registered off GPU, in section 2 of each counter arm's own plan. Its scorer is checked off GPU by scripts/dram_counter_route.py --self-test."
    return 0
  fi
  [[ -f "$lo" ]] || missing+=("$lo")
  [[ -f "$hi" ]] || missing+=("$hi")
  if (( ${#missing[@]} )); then
    skip_arm "$name" \
      "the contrast is a RATIO ACROSS two cells and ${missing[*]} was not written, so there is one cell and --analyse is what scores one cell. Read the counter pair's rows above for why: counter_plan not DONE retires both, and a run that exited before writing its payload is in its own log. Nothing is spent here and nothing is claimed."
    return 0
  fi
  arm "$name" "$PY_BASE" "$REPO/scripts/dram_counter_route.py" \
      --contrast "$lo" "$hi" --card "$COUNTER_PLAN_CARD"
}
counter_contrast_arm counter_contrast

# --------------------------------------------------------------------------
# Every arm's verdict, together, against the item it closes. THE ONLY THING
# GREPPED IS `^RESULT: `, printed verbatim: the arms' own scored-gate lines and
# nothing this driver worded itself.
# --------------------------------------------------------------------------
if (( DRY )); then
  say "THE PLANS, ARM BY ARM. NOTHING BELOW IS A MEASUREMENT."
else
  say "THE GATES, ARM BY ARM"
fi
for n in "${ARM_NAMES[@]}"; do
  wanted "$n" || continue
  log="$LOGS/$n.log"
  printf '\n--- %s ---\n' "$n"
  printf '  closes: %s\n' "$(arm_closes "$n")"
  state="$(awk -F'\t' -v a="$n" '$1==a{s=$2} END{print s}' "$LEDGER")"
  reason="$(awk -F'\t' -v a="$n" '$1==a{s=$7} END{print s}' "$LEDGER")"
  [[ -n "$state" ]] && printf '  state:  %s\n' "$state"
  [[ -n "$reason" ]] && printf '  reason: %s\n' "$reason"
  if summarize_arm "$n" "$log" "$state" "$reason"; then
    continue
  fi
  if [[ "$state" == "REFUSED" || "$state" == "PLAN_REFUSED" || "$state" == "NOT_PLANNED" ]]; then
    continue
  fi
  [[ -f "$log" ]] || continue
  if [[ "$state" == "BROKEN" ]]; then
    printf '  THE PLAN IS BROKEN: this arm exited %s under --dry-run, which is not\n' \
      "$(awk -F'\t' -v a="$n" '$1==a{r=$3} END{print r}' "$LEDGER")"
    printf '  a plan and not a refusal. Nothing above is a schedule for this arm.\n'
  elif (( DRY )); then
    printf '  PLAN ONLY: a --dry-run scores no gate, so it prints no RESULT line.\n'
    printf '  off GPU its gates are exercised by: %s\n' "$(arm_offgpu_gates "$n")"
  else
    printf '  NO `RESULT: ` LINE IN %s. This arm was NOT scored: a gate that\n' "$log"
    printf '  examined nothing reports no failures. Read the log before quoting\n'
    printf '  anything from it, and do not read prose in it as a verdict.\n'
  fi
  printf '  last 5 lines:\n'
  tail -5 "$log" | sed 's/^/    /'
done

say "ARMS"
cat "$LEDGER"
(( DRY )) || contract_disclosure "$LEDGER"
# THE ROWS WHOSE PAGE CONTRADICTS THEIR EXIT CODE, under their own heading, read
# off the ledger (last row per arm wins, as everywhere else) so a resume that
# did not touch a defective arm still shows it. A `RESULT: CLAIM C1 FAIL` page
# under exit 0 used to be latched DONE with nothing on this page to say so.
DEFECTS=""
OWED=""
if (( DRY == 0 )); then
  DEFECTS="$(defect_rows "$LEDGER")"
  OWED="$(owed_rows "$LEDGER")"
  if [[ -n "$DEFECTS" ]]; then
    say "THE ROWS WHOSE PAGE AND EXIT CODE DISAGREE"
    printf '  Each arm below printed RESULT lines that imply one exit code and then\n'
    printf '  returned another. moe/bench/exit_codes names that disagreement as itself\n'
    printf '  a defect, in either direction: a FAIL line under exit 0 is a refutation\n'
    printf '  filed as a pass, and all-PASS lines under exit 1 are a pass filed as a\n'
    printf '  refutation. Neither word is latched and neither may be quoted. The fix\n'
    printf '  belongs in the script, at the place it chooses its exit code; the\n'
    printf '  session exits INVALID over these rows, because what is on the page is\n'
    printf '  not a verdict.\n\n'
    printf '%s\n' "$DEFECTS"
  fi
  if [[ -n "$OWED" ]]; then
    say "THE ROWS THIS SESSION STILL OWES"
    printf '  Each arm below is UNKNOWN for a reason other than a defective page:\n'
    printf '  it exited 0 or 3 with no RESULT line (a check that examined nothing\n'
    printf '  reports no failures), or exited 1 from a file that has not adopted\n'
    printf '  the table, or left a log the second opinion could not read. No word\n'
    printf '  is latched, nothing from them may be quoted, and --resume-latest\n'
    printf '  re-attempts every one. The session exits INVALID over these rows,\n'
    printf '  because exit 0 was being read as "every arm produced a result".\n\n'
    printf '%s\n' "$OWED"
  fi
fi
printf '\ntotal %s min of wall clock\n' "$(( ($(date -u +%s) - started) / 60 ))"
printf 'work tree %s dirty file(s) at start, %s now\n' "$DIRTY_AT_START" "$(dirty_count)"

say "READ THESE FOUR FIRST"
cat <<EOF
  alias_ablation  the P1 line, and read it BEFORE the roofline verdict. It is
               the only arm that says whether alpha is a fraction of a DRAM
               weight read at all, which is the unit every roof fraction, every
               cap and the whole mechanism sentence is written in. READ THE P1
               RESULT LINE'S VERDICT WORD, NOT THIS ARM'S EXIT CODE: two of the
               four states below exit 1 and only one of them is a finding.
               P1 PASS: the relabelling stands and every number below keeps its
               subject. P1 FAIL, in that word: the per-tile slope is some other
               resource wearing DRAM's name, and the interval says which of
               0.10 or 0.33 it landed on instead. P1 UNKNOWN, whose detail
               opens NOT A REFUTATION, is not that, and since 2026-09-09 it is
               not what this arm spends its minutes on either: the arm is booked
               --dot-fallback refuse, so a probe that clears no sum-mode pinning
               stops there, for about 2.0 min and 3 INVALID, instead of running
               a dot ladder that measures a LOWER BOUND on alpha and cannot ask
               P1 at all. That is what the 2026-09-09 session bought under
               --dot-fallback allow: 308 s, P1 UNKNOWN at alpha >= 0.229, and a
               latched INVALID, four validity gates having failed after the
               money was spent. If the
               probe DOES clear, this is the P1 line the study has been waiting
               for. Do not write the 0.10-or-0.33
               sentence from an UNKNOWN. A headroom or attribution FAIL is none of the
               three: it is the apparatus saying it could not have seen DRAM
               whatever alpha is, which is what the 2026-09-01 attempt returned
               and what was nearly written up as a null result about DRAM.
  roofline-n256-g16  its REFUSAL, which is the arm's finding. This is the
               study's claim at the configuration vLLM ships (BLOCK_M=128,
               BN=256, G=16) and it CANNOT be confirmed on sm_90: the BLOCK_M=256
               control is refused at every warp and stage count (65536 of
               65536 registers per block), no BLOCK_SIZE_N confirms the
               headline on this card, and nothing else in this session can
               supply a control. Read roofline-n64-g1 beside it as the control
               it is: it can only refute. Were a card ever to run the control,
               CEILING BINDING AT THE PRODUCTION TILE would confirm the claim
               and CEILING NOT BINDING would refute it; on this card neither
               line is printed and the paper has no confirming arm.
  noise_floor  the between-replicate sd. Every effect anywhere in this study is
               to be read against it from now on, and until it exists the MDE
               printed at the top of this run is an ASSUMPTION carried from a
               same-session, same-hour, mixtral-only proxy that confounds
               num_stages.
  bn_g16       the residual line. Noise-sized residual = the three-term model is
               complete. Structure in it names what is missing. It is the ONLY
               bn arm now: at GROUP_SIZE_M=1 that script's own design self-test
               exits 3 INVALID and its plan says C1 reads UNKNOWN however the
               data fall, and no other pinning it was checked at passes either.
EOF

# THE NEXT SESSION, printed only where a session was actually run: a --dry-run
# has no ledger of results to book a rerun against, and printing a rerun plan
# under a plan would be a second schedule on the same page.
if (( DRY == 0 )); then
  next_session_booking
fi

say "WHAT TO COMMIT, AND WHAT NOT TO"
cat <<EOF
  NOTHING HERE IS COMMITTED AND NOTHING IS PUSHED. This script runs no git write
  command and terminates nothing. Read the gates above first.

  THE TWO FILES THAT BELONG IN THE REPO are this card's calibration and this
  session's noise floor. Both are written by an arm that was given --publish,
  and both are tracked:
      git -C $REPO diff --stat moe/bench/hardware/ results/published/NOISE_FLOOR.json
  Do not commit the calibration if arm 9 (ruler) C1 PASSED. C1 is
  "the GEMM clock was sampled in the wrong state", gated at a post-hoc-versus-
  under-load delta above 5%, so a PASS is the arm SUSTAINING that charge: the
  compute peak was read in one clock state and the ridge derived from it is not
  this card's. A C1 FAIL is the designed null and the state in which the
  calibration is committable; on 2026-09-09 it read FAIL at a measured delta of
  0.0% on both dtypes. TWO THINGS WERE WRONG WITH THIS LINE UNTIL 2026-09-09.
  It named P1, and ruler_rebaseline.py scores V1-V3 and C1-C5 with no P1 at
  all, so it sent the operator to a gate that is not on the page. And it read
  the verdict backwards, telling them to withhold the calibration in exactly
  the state that clears it. Read the measured delta on the C1 line either way.
  Do not commit NOISE_FLOOR.json if the noise_floor arm is not DONE. Its V2 asks
  that the PLANNED replicates ran, so a floor from an arm cut short is not a
  floor, and this run bought N=3, at which the estimate is known to 1.92x -- the
  scope block in the file says so and must not be quoted away from it. Committing
  it is what finally replaces the assumed sigma of 0.0228 in every later
  session's detection-limit line with a number this study measured.

  EVERYTHING ELSE LANDS UNDER $RESULTS, which git ignores by design. To publish
  an arm, copy its run directory under results/published/<date>-<gpu>-<arm>/ --
  the report.json files are kilobytes -- then:
      git -C $REPO check-ignore -v <the path you chose>     # must print NOTHING
  Three published directories were silently dropped by results/* this week
  before that check was made routine.

  THE ALIAS ABLATION WRITES NOTHING TRACKED AND IS THE ARM MOST WORTH
  PUBLISHING. It has no --publish flag because it has no tracked file to land
  in: report.md, plan.json, cells.jsonl and probe.json go under
  $RESULTS/alias_ablation/<run id>, which git ignores. Copy that directory into
  results/published/ whichever way its P1 line reads. A PASS and a FAIL are both
  results and both decide the paper's mechanism sentence. An INVALID on headroom
  or attribution is worth publishing too, and as an APPARATUS finding rather
  than as a fact about the card: that is exactly what the 2026-09-01 run was,
  and its VOID was nearly read as a null result about DRAM.

  THE SESSION DIRECTORY IS $SESSION ($SESSION_HOW).
  Its ledger is the only place the latch lives. To resume it, skipping every
  finished arm, run one of:
      bash scripts/h200_gaps_session.sh --resume-latest
      SESSION=$SESSION bash scripts/h200_gaps_session.sh
  A bare re-invocation on this card now REFUSES rather than opening an empty
  ledger beside this one; --new opens one on purpose.

  A RESUME RE-RUNS NO INVALID ROW AND NO CLAIM_FAIL ROW. That is the latch
  working as designed, not a bug to route around: both are RESULTS, both are
  latched, and \`arm\` skips any arm already holding one. So after a session
  whose arms landed INVALID (the 2026-09-09 set landed six), --resume-latest
  runs nothing for them however many defects have been fixed since. The next
  session for those arms is --new, which opens a fresh ledger on purpose. A
  measuring run prints the exact command and the state each arm is expected to
  reach, a few lines above this one, under THE NEXT SESSION, AND WHY IT IS
  --new; a --dry-run has no results to book a rerun against and prints no such
  block. Deleting rows out of a ledger by hand is the other way back and it
  edits the record of what was spent; --new does not.
  Copy it off before releasing the pod:
      tar czf /workspace/exfil-gaps-$CARD.tar.gz -C "$(dirname "$SESSION")" "$(basename "$SESSION")"
EOF

# The session's own exit code, by the same rule it applies to its arms: a plan
# that did not survive its own --dry-run is not a plan, and an arm that exited a
# code nobody chose has not reached a state. Neither is a claim, so neither is 1.
if (( DRY )) && (( BROKEN_ARMS > 0 )); then
  say "$BROKEN_ARMS PLAN(S) BROKEN. The plan above is not a plan; fix them before the pod."
  awk -F'\t' '$2 == "BROKEN" { printf "  %-19s exited %s under --dry-run   %s\n", $1, $3, $6 }' "$LEDGER"
  cat <<'EOF'
  A plan may exit 0 (PLANNED) or 2 (PLAN_REFUSED, the refusal working off a GPU
  box). Any other code is a plan that did not print. An arm here exiting 3 is
  almost always the exit-code contract this session adopted and its script has
  not: 3 is INVALID, "measured and then a VALIDITY gate failed", and a
  --dry-run measures nothing, so a refusal that exits 3 is a refusal wearing an
  INVALID's number. Fix it in the script, through moe/bench/exit_codes.classify
  and REFUSED, not here.
EOF
  exit "$RC_INVALID"
fi
if (( DRY == 0 )); then
  SESSION_RC="$(session_rc "$LEDGER" "$RETRY_ARMS")"
  if [[ -n "$DEFECTS" ]]; then
    say "$(printf '%s\n' "$DEFECTS" | wc -l | tr -d ' ') ROW(S) WHOSE PAGE AND EXIT CODE DISAGREE. Nothing from them may be quoted."
  fi
  if [[ -n "$OWED" ]]; then
    say "$(printf '%s\n' "$OWED" | wc -l | tr -d ' ') ROW(S) STILL OWED: UNKNOWN, not latched, not a result. --resume-latest re-attempts them."
  fi
  if (( SESSION_RC == RC_RETRY )); then
    say "$RETRY_ARMS ARM(S) EXITED A CODE OUTSIDE THE TABLE, OR CRASHED BEFORE SCORING A GATE. Read their logs."
  fi
  exit "$SESSION_RC"
fi
exit 0
