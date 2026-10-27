#!/usr/bin/env bash
# THE GH200 MODEL-TEST SESSION, RUN UNATTENDED ON THE VM: THE CALIBRATION RENTAL C3'S DRIVER (2026-10-26).
#
#   bash gh200_cal3_session.sh --dry-run --plan scripts/plans/<file>.plan   # the plan; exit 2
#   bash gh200_cal3_session.sh --setup --commit <sha> --repo <url> --deadline <epoch s> --plan ...
#   bash gh200_cal3_session.sh --plan scripts/plans/<file>.plan --deadline <epoch s>
#   (scripts/vm_run.sh start ... --driver gh200_cal3_session.sh)
#
# THIS FILE IS scripts/gh200_cal2_session.sh AS PINNED BY THE CALIBRATION RENTAL C2 (docs/registered/
# 2026-10-22-cal2-gh200 code_pins) WITH THE CALIBRATION RENTAL C3'S CHANGES (docs/registered/2026-10-26-cal3-gh200).
# The C2 driver stays byte-identical; every line this copy adds or changes past this header is marked C3 (a `# C3`
# comment, or `$(: C3)` on a line continued with a backslash) and tests/test_cal3_plan.py holds every other line to
# the C2 driver. C3 is a CALIBRATION rental, DESCRIPTIVE (it certifies nothing; its pages are SEEN). It keeps C2's
# prelude (700 W read back on both readings or STOP, the 1410 lock read back or STOP) and A2's G-BAND gate unit, and
# runs the instrumented copy of vLLM's kernel (moe/instrumented, unchanged) at vLLM's own 128x128 tile:
#   the probe  every instrumented step (regcheck, perturb, stamps) runs scripts/instr_probe_c3.py, rental 4/5's probe
#              made for vLLM's 128x128x64 w8 s3 G 1 tile and A2's E1-CAL histogram cells (scripts/instr_probe.py,
#              pinned, is imported for its helpers and stays byte-identical)
#   stamps     a stamps unit may name a histogram page (histogram=, a registered file) with a whole tile (block-m,
#              block-n, block-k, num-warps, num-stages); such a unit takes no stamp-treads (its treads are its
#              cells') and needs a regcheck unit before it; instr-stamps=ends (the CTA ends only, nothing near the
#              wgmma k-loop) joins cta and iter. The stamps step hands the probe the regcheck file as well as the
#              gate file: the unit runs only when both read PASS (decision 2; rental 5's regcheck)
#   variants   a histogram stamps unit's variant (instr-variants.json) carries its tile and histogram file
#   OD-1       (owner, 2026-10-26) a histogram stamps unit runs on the perturb unit's HARD line (every gate leg but
#              timing: occupancy, the all-off SASS, the upstream file, C3's SASS legs); a FAIL on the timing legs alone
#              keeps the unit, and the probe labels its pages PERTURBED
#
# THE C2 DRIVER'S OWN HEADER, UNCHANGED:
# THE GH200 MODEL-TEST SESSION, RUN UNATTENDED ON THE VM: THE CALIBRATION RENTAL C2'S DRIVER (2026-10-22).
#
#   bash gh200_cal2_session.sh --dry-run --plan scripts/plans/<file>.plan   # the plan; exit 2
#   bash gh200_cal2_session.sh --setup --commit <sha> --repo <url> --deadline <epoch s> --plan ...
#   bash gh200_cal2_session.sh --plan scripts/plans/<file>.plan --deadline <epoch s>
#   (scripts/vm_run.sh start ... --driver gh200_cal2_session.sh)
#
# THIS FILE IS scripts/gh200_settleS1_session.sh AS PINNED BY THE SETTLE RENTAL S1 (docs/registered/
# 2026-10-21-settleS1-gh200 code_pins) WITH THE CALIBRATION RENTAL C2'S CHANGES (docs/registered/2026-10-22-cal2-gh200).
# The S1 driver stays byte-identical; every line this copy adds or changes past this header is marked C2 (a `# C2`
# comment, or `$(: C2)` on a line continued with a backslash) and tests/test_cal2_plan.py holds every other line to
# the S1 driver. C2 is a CALIBRATION rental (it certifies nothing; its pages are SEEN for v5'' and B3). Its changes:
#   prelude    the plan's limit only (700 W for C2: S1's publication showed -pl 900 -sc 0 leaves the enforced GPU
#              limit at 700 W, so 900 W is not settable this way): S1's read-back (enforced.power.limit and -q -d
#              POWER's GPU-scope Current Power Limit both at the limit within 1 W, polled up to 30 s), and on a
#              refusal the session STOPS: no fallback limit (S1's 900 to 700 W fallback is not taken). c2-power.env
#              records C2_POWER_W, or C2_POWER_<W>=refused.
#   the ledger S1's defect fixed: a limit that never reads back leaves its last reading in
#              power-set-unsettled-<W>.txt, and the ledger line names that file (S1's named power-set-unsettled.txt,
#              a file its prelude then renamed to power-set-unsettled-900.txt).
#   lock       the lock-hold probe in its cheapest form, S1's prelude probe made a stop: the prelude locks the
#              session lock, reads clocks.sm back (lock_and_check, two tries), resets, and records
#              LOCK_PROBE_<L>=held; a lock that does not read back REFUSES the prelude, before any page. The plan
#              adds A2's G-BAND gate (tileA2_gate.py band) after the two f128 pages: their counter clocks under load
#              must sit in the registered band at the lock, else the session ends there.
#   the ledger each unit's START line and the dry run name the plan prelude's power limit (S1's printed the
#              driver's default 900 W whatever the plan's power-limit-w).
#
# THE S1 DRIVER'S OWN HEADER, UNCHANGED:
# THIS FILE IS scripts/gh200_tileA2_session.sh AS PINNED BY TILE RENTAL A2 (docs/registered/2026-10-20-tileA2-*
# code_pins) WITH THE SETTLE RENTAL S1'S CHANGES (docs/registered/2026-10-21-settleS1-gh200). The A2 driver stays
# byte-identical; every line this copy adds or changes past this header is marked S1 (a `# S1` comment, or
# `$(: S1)` on a line continued with a backslash) and tests/test_settleS1_plan.py holds every other line to the A2
# driver. S1 is DESCRIPTIVE (it measures, it certifies nothing; no gate, no lock-hold probe). Its changes:
#   prelude    the power read-back (A2's defect: power-set.txt was written right after `nvidia-smi -pl` and A2's
#              -q -d POWER read 700 W while power.limit read 900 W for the whole session): after -pl 900 -sc 0 (scope 0
#              is the GPU's) the prelude polls, every MOE_DRIVER_PL_POLL_S (2 s) up to MOE_DRIVER_PL_WAIT_S (30 s),
#              until --query-gpu=enforced.power.limit (the minimum of every limiter) and -q -d POWER's GPU-scope
#              Current Power Limit both read the plan's limit within 1 W; only then are power-set.txt and
#              clocks-at-start.txt written. Otherwise the prelude REFUSES (the last reading is power-set-unsettled.txt).
#              A refusal at 900 W is recorded (s1-power.env, power-set-unsettled-900.txt) and the SAME session
#              sets 700 W and repeats the read-back (owner, 2026-10-21; no other scope is tried); every unit then runs
#              at 700 W. A refusal at 700 W too stops the session. s1-power.env S1_POWER_W names the limit in force.
#   powerproof (a step, `- powerproof label=...`): S1-M0, scripts/power_proof_s1.py's sustained bf16 GEMM for 90 s
#              with no clock lock while nvidia-smi logs power.draw, enforced.power.limit and clocks_event_reasons
#              every 100 ms: the draw shows which limit held.
#   the ledger each unit's START line names the lock the unit runs at (its own lock-mhz, else the session's
#              a2-lock.env, else the plan prelude's); A2's printed the default 1710 for every unit.
#   lock-mhz=F on a bytes unit: that unit's own lock (S1-M2's 1305 and 1005 MHz pages), probed by the prelude.
#   byte-treads=n,n (bytes units without histogram=): the ladder's treads (R3's --tiles), in place of 1..9
#              (S1-M4's PRIVATE pages at the cells A2's byte pages measured).
#   replay=application (bytes units with histogram=): the page through scripts/dram_counter_route_s1.py
#              --s1-replay application, ncu's application replay with no cache flush (S1-M1).
#   stall-metrics=1 (bytes units with histogram=): the page through scripts/dram_counter_route_s1.py
#              --s1-stall-metrics, three stall and pipe metrics recorded per GEMM (S1-M5).
#
# THE A2 DRIVER'S OWN HEADER, UNCHANGED:
# THIS FILE IS scripts/gh200_tile_session.sh AS PINNED BY THE TILE STAGE (docs/registered/2026-10-14-tileA-*
# code_pins) WITH TILE RENTAL A2'S CHANGES (docs/registered/2026-10-20-tileA2-gates-gh200). The tile driver
# stays byte-identical; every line this copy adds or changes past this header is marked A2 (a `# A2`
# comment, or `$(: A2)` on a line continued with a backslash) and tests/test_tileA2_plan.py holds every
# other line to the tile driver. A2 (owner decisions 2026-10-20): a 900 W GPU power limit, which changes the
# owner's 2026-09-26 700 W decision (below) for A2 (and rental B2, at the limit A2 held) only, and the lock
# ladder 1710, 1410, 1305: the first rung whose probe passes is the lock of every later unit.
#   prelude    keys lock-mhz=L (the session's lock, a2-lock.env) and power-limit-w=700|900. The prelude
#              reads the card's power.max_limit FIRST: under the plan's limit it REFUSES (the session
#              stops before any unit; never a fallback to 700 W); then -pl W -sc 0 and the read-back.
#   lockprobe  (a step, `- lockprobe label=... ladder=1710,1410,1305 probes=M:FILE:G:s,...`, the tile keys):
#              per rung L, hw_rulers at L, then each probe page timed at L (locked_r3, slip-policy cell,
#              r7-median-run3), the module power sampled, then scripts/tileA2_gate.py probe; the first rung
#              that HOLDS is the session's lock (a2-lock.env A2_LOCK); none holds: the session STOPS.
#   gate       (a step, `- gate label=... gate=calibrate|census|band|f64w8`): scripts/tileA2_gate.py; any
#              exit but 0, or a gate skipped for time, STOPS the session (card reset, results pushed).
#   lock-mhz=F on a timed or rulers unit: that unit's own lock (CONV-L's 1710 twin, the DESCRIPTIVE 1710
#              probe and rulers);
#              the prelude probes every such lock and every ladder rung.
#   unless-lock=F on a timed unit: the unit is skipped (exit 0, ledgered) when the session's lock is F
#              (CONV-L: no conversion to test when rung 1710 holds).
#   --dry-run --plan on a box with nvidia-smi reads power.max_limit first: under the prelude's
#              power-limit-w it prints REFUSED and no plan, so vm_run.sh start stops before setup.
#   tilecensus runs scripts/tile_census_a2.py (each model at its smallest whole tread; deepseek-v2-lite
#              at 3) instead of the pinned tile_census.py.
# THIS FILE IS scripts/gh200_model_session.sh AS PINNED BY RENTAL 7 (registered sha256
# 2026-10-13-rental7-*-gh200 code_pins) WITH THE TILE STAGE'S KEYS ADDED. The pinned driver stays
# byte-identical (a registered code pin is never edited); every line this copy adds or changes is
# marked TILE (a `# TILE` comment, or `$(: TILE)`, an empty word, on a line continued with a backslash)
# and tests/test_tile_plan.py holds every other line to the pinned driver.
# THE TILE STAGE (2026-10-14; design-tile 5, design-tile-review 6). Three keys and one step.
#   block-m=BM block-n=BN num-warps=W (timed and bytes units; a unit naming one names all three, with
#              block-k and num-stages): vLLM's config values for the unit's pages. R3 and the counter
#              route are pinned and have no warps flag, so a unit with num-warps runs R3 through
#              scripts/r3_tile.py (locked_r3's existing --r3-script; the dry run too) and its byte
#              pages through scripts/dram_counter_route_tile.py, each of which sets the sweep's FIXED
#              num_warps in memory, records it on the page and keys the run ids by it.
#   tilecensus (a step, `- tilecensus label=... tiles=T,T models=M,M`): scripts/tile_census.py,
#              compile only (no lock, nothing timed): every tile's registers, shared, CTAs per SM
#              and main-loop PTX counts on every named model, refused (exit 3) on a path mismatch.
#
# WHAT IT TESTS. The two NOT-FINAL models of docs/COUNTERS.md 6.7 on a GH200
# (`gpu_1x_gh200`): the wave-split byte model (scripts/wave_split_bytes.py) and
# the timing model LOWM-1/2 (scripts/r3_timing_model.py, predictions P1 to P8).
# The duty chain cannot hold this card's clock still (section 3b, the record of
# 2026-09-25: V7 and V0 failed under duty), so every timed page runs under an
# nvidia-smi lock through scripts/locked_r3.py.
#
# THE OWNER'S DECISIONS, 2026-09-26: byte pages count every tread from 1 to 9;
# calibrate runs on the VM after the byte pages; the GPU power limit is set to
# 700 W at the start (-sc 0), the limit the 2026-09-25 lock pages ran at; a VM
# that ships r570 is upgraded to 580.105.08 before setup_vm.sh
# (scripts/vm_run.sh does that, not this script).
#
# THE STEPS, in the owner's priority order, so a run cut short has run the
# earlier ones. Minutes are estimates from the 2026-09-25 logs and the tools'
# own plans; each step also has a cap, past which it is stopped:
#
#   prelude    3c.1  supported clocks, persistence, 700 W        the locks the later steps may use
#   bytes      3c.2  byte pages at the 1710 lock, G = 1 2 4 16 8 32 3 64, treads 1-9,
#                    and one base-clock control                  the byte predictions; counted bytes for every timed G
#   calibrate  3c.3  this card's ruler, with no lock in force   the ruler R3 needs
#   timed      3c.3  timed R3 at 1710: G=8, 32 (treads 6), G=3 (treads 8)   P2, P5
#   eta        3c.4  timed R3 at held locks 1410 (G=4, 2), 1500 and 1605 (G=4)   P1
#   floor      3c.5  floor counters: G=64, G=2 at ncu's base clock, G=64 unlocked
#                    (a record), G=64 under the 1710 lock        P6
#   deep       3c.6  timed R3 at 1710 to tread 9, G=4 then G=2  P3, P4
#   r1lock     3c.7  R1 in lock mode at 1710 1500 1410, G=4 then G=1   the floor's clock exponent
#   launchfloor      scripts/launch_floor.py at the 1710 lock (2026-10-01): R3's eager timer
#                    against a graph replay, flush levers and a profiler trace. In no default
#                    plan: a --steps list or a --plan line asks for it
#
# A PLAN OF SEVERAL MODELS (2026-10-01, `--plan FILE`): one rental runs units in the
# file's order, one per line, `<model> <step> [key=value ...]` (keys: label,
# byte-groups, floor-groups, floor-treads, lf-treads, lf-modes, lf-trace, est, cap;
# rental 2's bytes keys num-stages, block-k, slot-pad-rows, partition-metrics=1 and
# floor keys floor-lock-mhz, floor-base=0, floor-shape-metrics=1, floor-null-kernel=1,
# each refused on a unit of another step; drop-group=NAME, units the deadline drops
# together; `#` comments). The first unit is `- prelude`.
# RENTAL 3 (2026-10-05): a plan also runs calibrate, timed and deep, each with a label=.
# Their R3 pages go under $RESULTS_ROOT/gaps-<card>-<model>-<label> (a timed and a deep unit
# of one label share that directory by design, under different session tags); the ruler
# stays ONE per card (calibrate's measured_<card>.yaml and ruler.env), so a timed unit with
# no calibrate of its own (the rep8 replicate) times on the ruler the e2e calibrate set. New
# keys: timed-groups=G[,G] and timed-treads=N (timed units: those groups at those treads,
# and no G=3 page); lf-arms=<arms>, lf-group-m=N and lf-trace=0 (launchfloor units: the
# G = 64 NATIVE wall-time check, timed phase only, outside ncu). Each unit's step runs with that
# model and those overrides in the environment (MOE_DRIVER_*), its pages in a
# directory of its own (`<date>-<card>-<model>-<label>-r3-counters`, or
# `-launch-floor-<label>/<model>` for launchfloor), so no two page groups mix in one
# load_card or overwrite each other, and each unit is pushed as it lands. Before
# each unit, while the estimates still to run overrun the deadline less
# RESERVE_MIN, the LAST unit not yet run is dropped (the file's order is the
# priority order). `--dry-run --plan FILE` prints every unit.
# RENTAL 4 (2026-10-06): four plan-only steps and their keys, each refused off its step.
#   rulers     scripts/hw_rulers.py at the 1710 lock: torch read / copy / add / matmul rulers
#              and the device's L2 size (eta_mix = bw / ruler on this board)
#   gpubench   scripts/gpubench.py at the 1710 lock: THIRD-PARTY RRZE-HPC/gpu-benches (GPL-3.0,
#              owner decision 3), fetched at its pinned commit into $MOE_HOME/ext (outside the
#              checkout), built with nvcc, latency / stream / l2-cache run once each
#   perturb    scripts/instr_probe.py --mode perturb at the 1710 lock: the owner's decision-2
#              gate (median |plain / copy - 1| <= 1%, worst <= 2%, identical occupancy) for
#              EVERY instrumented unit after it in the plan; writes $D/instr-gate.env
#   stamps     scripts/instr_probe.py --mode stamps at the 1710 lock: the instrumented copy of
#              vLLM's kernel (moe/instrumented) with timestamps on the unit's cells; keys
#              stamp-groups, stamp-treads, stamp-arms, instr-stamps=cta|iter, instr-every,
#              instr-marks, block-k, num-stages. REFUSED unless its gate line reads PASS
#   keys       declared-copies=N (timed units ONLY: R3's --declared-copies, refused on bytes and
#              floor units, whose counter pages declare R3's 9; the page's directory carries a
#              NOT-JOINABLE label when N is not 9); instr-evict-a / instr-evict-b=first|last|none
#              (bytes units: the copy's eviction hints on a counter page, gated like stamps;
#              the gate FAILS unless the compiled PTX carries L2::cache_hint, which Triton
#              3.7.1 drops on pipelined loads, so rental 4's plan has no such unit)
# Every instrumented unit needs a perturb unit before it in the file. The plain installed
# kernel stays the measured object of every timed, deep, floor and plain bytes unit.
# RENTAL 5 (2026-10-07): one plan-only step and these keys, each refused off its step.
#   regcheck   scripts/instr_probe.py --mode regcheck, compile only (no lock, nothing timed):
#              every variant of the plan's instrumented units at the plain kernel's CTAs per SM
#              (registers printed); writes $D/instr-regcheck.env. It runs right after
#              calibrate, before any timed unit (owner decision 2): a plan with a regcheck unit
#              after a timed or deep unit is refused
#   keys       histogram=FILE (timed and bytes units: R3's / the counter route's --histogram, a
#              histogram page of docs/registered/2026-10-07-rental5-skew-hist; a timed unit needs
#              declared-copies), arms=native[,shared] (with histogram= only: R3 refuses --arms on
#              a ladder page; PRIVATE never runs skewed), block-k / num-stages on timed units too
#              (R3's --block-k / --num-stages), instr-variants=v1[,v2[,v3]] (regcheck, perturb and
#              stamps units: the registered stamp variants, moe.instrumented.R5_VARIANTS; a stamps
#              unit runs the FIRST of its variants whose regcheck and perturb lines both read PASS
#              and writes CHOSEN_VARIANT.txt), depends=GROUP (a unit dropped when any unit of
#              drop-group GROUP is dropped: rental 5's stT and stD depend on st)
# RENTAL 6 (2026-10-09): two keys.
#   slip-policy=page|cell (timed units with histogram= only): locked_r3.py's --slip-policy.
#              cell: a slipped repeat no longer stops the page; R3 flags it, keeps it out of
#              the cell median, drops a cell under 6 clean repeats of 9 and stops the page once
#              none can reach 6; the page ends HELD_WITH_SLIPS (exit 0). page is the default.
#   nodrop=1   (any unit but the prelude, which is never dropped anyway): the deadline never
#              drops this unit, and no drop-group sweeps it; refused with drop-group= or
#              depends=. When dropping every droppable unit still leaves too little time, the
#              nodrop units are still not dropped: each runs while its own estimate fits the
#              time left (a unit that no longer fits is SKIPPED, as any step), and the ledger
#              says so (rental 6's nine skew pages: design-r6-review (g)).
# RENTAL 7 (2026-10-13): one key, and the log names.
#   g1-rule=r5-any|r7-median-run3 (timed units with histogram= only; r7-median-run3 also needs
#              slip-policy=cell). r7-median-run3 is the SCORER's rule (registration
#              2026-10-13-rental7-g1rule, scripts/scoring/rental7/r7common.py, rebuilt from cells.csv);
#              R3 and locked_r3 are rental 6's, byte-identical (registered pins), and get no new flag.
#              The key hands locked_r3 --min-clean-repeats 0, so R3 times every repeat of every cell
#              and never stops a cell or the page early (under the default 6 its r5-any LOST rule would
#              stop a cell the scorer's rule keeps: design build review, about 19% of Phi n 1-2 NATIVE
#              cells truncated at a 30% flag rate); the unit's R3 dry run carries the slip arguments
#              locked_r3 then adds (--slip-policy cell --lock-mhz F --min-clean-repeats 0). r5-any, or
#              no key: as before.
#   logs       every per-unit log under $S/logs carries the unit's model and label (uid below).
#
# WHY UNATTENDED, AND HOW THE DRAFT'S "STOPS" BECAME RULES. The draft of
# section 3c (2026-09-26) was blocks a person pastes, each with a "Stops"
# paragraph read before the next block. On an unattended VM every one of those
# readings is a rule here, and a stop that needs a person stops only its own
# step: the driver resets both kinds of lock, pushes what the step wrote, and
# goes on to the next step, because every step checks the card for itself
# before it measures (idle, no lock left, its own dry run). Three things stop
# the whole session instead, because nothing after them could be trusted: the
# prelude refusing (a dirty checkout, a busy card, no 1710 lock, persistence or
# 700 W not taken), a reset that fails (`nvidia-smi -rgc` exiting non-zero:
# the card may still be locked), and a card still busy five minutes after a
# step ended. The rules, step by step, are in docs/LAMBDA.md section 3c.
#
# RESULTS GO TO GITHUB AS THEY LAND. After every step and every byte page, and
# only between them (never while a page is measured), scripts/vm_results_push.sh pushes the VM's
# results and session to the run's own branch. The laptop reads them with `git
# fetch`, and terminates the instance once DRIVER-DONE is on the branch.
#
# TIME. --deadline is the epoch second by which the driver must have ended,
# final push included (the laptop sets it: launch + 400 minutes, for a 7-hour
# stop). Before each step the driver adds the estimates of every step still to
# run; while that overruns the deadline less RESERVE_MIN it drops, in the
# draft's order, R1 at G=1, then R1 at G=4, then the 1605 lock, then the G=64
# byte page and the base-clock control. A step whose own estimate no longer
# fits is skipped. A step's cap never runs past the deadline.
#
# EXIT CODES, the repo's table (moe/bench/exit_codes.py): 0 every step ran and
# ended 0 or 1; 2 refused before measuring (the prelude, or bad arguments; and
# --dry-run); 3 a step stopped short, was skipped or ran out of time; 4 a lock
# reset failed, the card stayed busy, or a step crashed.
#
# PRIVACY. locked_r3.py's logs and ledger, and every page, carry the card's
# UUID. This driver's own ledger ($SESSION_ROOT/gh200-driver/status) copies
# only the ledger lines that name no card (`G=<n> <state> at lock <F>: R3 exit`)
# and never a `card:` line, so it can be read whole.
#
# TEST SEAMS: MOE_HOME (default $HOME/moe: env.sh, session, results),
# MOE_DRIVER_RESERVE_MIN, MOE_DRIVER_BUSY_WAIT_S, MOE_DRIVER_NOW (a fixed
# clock, epoch seconds, for the budget tests), MOE_DRIVER_REPRICE_S (the G=1
# page's re-price threshold, 420 s).
set -uo pipefail

EXIT_DONE=0
EXIT_CLAIM_FAIL=1
EXIT_REFUSED=2
EXIT_INVALID=3
EXIT_ERROR=4

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
MOE_HOME="${MOE_HOME:-$HOME/moe}"
ENV_SH="$MOE_HOME/env.sh"

# ---- the design: the owner's decisions and the registered predictions ------
#: The lock every timed page and every byte page runs at: P2 to P6 are
#: registered at it, and the 2026-09-25 pages ran at it.
LOCK_TIMED=1710
#: The owner's GPU power limit (2026-09-26), GPU scope only (-sc 0). On a GH200
#: the module's limit (-sc 1) is shared with the Grace CPU; both are recorded.
POWER_LIMIT_W="${MOE_DRIVER_POWER_LIMIT_W:-900}"   # A2: 900 W for A2 (owner, 2026-10-20; the 2026-09-26 700 W elsewhere)
#: Byte pages count every tread from 1 to 9: private_weight_reference.py's
#: COUNTER_MAX_TREADS (a test holds this to it).
TREADS=1,2,3,4,5,6,7,8,9
#: The owner's six, then 3 (the timed G=3 page's bytes), then 64 (dropped first).
BYTE_GS=(1 2 4 16 8 32 3 64)
#: The first byte page re-prices the rest: past this many seconds, the G=64
#: page and the base-clock control are dropped (the draft's 7 minutes).
BYTE_G1_REPRICE_S="${MOE_DRIVER_REPRICE_S:-420}"
#: P1's registered held locks, scripts/r3_timing_model.py P1_CLOCKS (a test
#: holds these to it). 1605 falls back to 1590, whose P1 numbers were
#: registered with 1605's; any other substitute is labelled as computed after
#: registration.
P1_LOCKS=(1410 1500 1605)
P1_FALLBACK_1605=1590
#: The 2026-09-25 GH200 ruler's numbers: R3 takes them when calibrate fails.
RULER_FALLBACK=(--ridge 177.93 --bandwidth-gbps 3725.1)
#: R3's design for every timed page (the chain's, as on 2026-09-25).
#: The model R3, R1 and every counter page run (--model; the steps read it
#: from MOE_DRIVER_MODEL, which the orchestrator exports). Mixtral 8x7B is the
#: study's; another model is a cross-model test of the two per-card models
#: (docs/registered/README.md), with its own census and no 8x7B references.
DEFAULT_MODEL=mixtral-8x7b
#: The counter pages' declaration (private_weight_reference.counter_declaration
#: over its 9-deep ladder): 9 copies for every model the driver runs.
COUNTER_COPIES=9
MODEL="${MOE_DRIVER_MODEL:-$DEFAULT_MODEL}"
#: A registered floor design (2026-09-30): --floor-groups G,G and --floor-treads n,n,...
#: replace the floor step's layout (G=64, the model's r3_floor_treads, the r3f-g2 and
#: unlocked records) with those groups at those treads, each at ncu's base clock and at the
#: lock. Empty: the default layout, unchanged. The steps read them from the environment.
FLOOR_GROUPS="${MOE_DRIVER_FLOOR_GROUPS:-}"
FLOOR_TREADS="${MOE_DRIVER_FLOOR_TREADS:-}"
#: A registered byte-page list (2026-10-01): --byte-groups G,G,... replaces BYTE_GS for
#: the bytes step, the floor override's pattern. It is the registered list: nothing of
#: it is dropped or re-priced, and no base-clock control or C5 reference is taken (they
#: compare clock regimes and the 2026-09-25 timed pages, which no such list reads).
BYTE_GROUPS="${MOE_DRIVER_BYTE_GROUPS:-}"
#: A plan unit's own name for its pages' directory (see A PLAN OF SEVERAL MODELS).
#: Empty: the session's one counters directory, as before.
OUT_LABEL="${MOE_DRIVER_OUT_LABEL:-}"
#: RENTAL 3 (2026-10-05): a timed unit's own groups and treads (timed-groups, timed-treads).
#: Empty: G=8, 32 at treads 6 and G=3 at treads 8, as before. Either given: those groups
#: (8 32 by default) at those treads (6 by default), and no G=3 page.
TIMED_GROUPS="${MOE_DRIVER_TIMED_GROUPS:-}"
TIMED_TREADS="${MOE_DRIVER_TIMED_TREADS:-}"
#: RENTAL 4 (2026-10-06): a timed unit's declaration (declared-copies); empty: COUNTER_COPIES
#: for a non-Mixtral model and R3's own rule (9) for Mixtral, as before.
DECLARED_COPIES="${MOE_DRIVER_DECLARED_COPIES:-}"
#: the instrumented copy's keys (moe/instrumented): a bytes unit's eviction hints, a stamps
#: unit's stamps, cells and config; the unit's variant id in $D/instr-variants.json
INSTR_EVICT_A="${MOE_DRIVER_INSTR_EVICT_A:-}"
INSTR_EVICT_B="${MOE_DRIVER_INSTR_EVICT_B:-}"
INSTR_STAMPS="${MOE_DRIVER_INSTR_STAMPS:-}"
INSTR_EVERY="${MOE_DRIVER_INSTR_EVERY:-}"
INSTR_MARKS="${MOE_DRIVER_INSTR_MARKS:-}"
STAMP_GROUPS="${MOE_DRIVER_STAMP_GROUPS:-}"
STAMP_TREADS="${MOE_DRIVER_STAMP_TREADS:-}"
STAMP_ARMS="${MOE_DRIVER_STAMP_ARMS:-}"
VARIANT_ID="${MOE_DRIVER_VARIANT_ID:-}"
#: RENTAL 5 (2026-10-07): a histogram page (timed and bytes units), its arms, and a stamps
#: unit's registered variants (instr-variants) with their ids in preference order
HISTOGRAM="${MOE_DRIVER_HISTOGRAM:-}"
R3_ARMS="${MOE_DRIVER_ARMS:-}"
#: RENTAL 6 (2026-10-09): a timed unit's slip policy (slip-policy); empty: locked_r3's page
SLIP_POLICY="${MOE_DRIVER_SLIP_POLICY:-}"
#: RENTAL 7 (2026-10-13): a timed unit's scorer-side G1 rule (g1-rule); r7-median-run3: min-clean 0
G1_RULE="${MOE_DRIVER_G1_RULE:-}"
#: THE TILE STAGE (2026-10-14): a unit's tile and a tilecensus unit's tiles and models   # TILE
TILE_BLOCK_M="${MOE_DRIVER_BLOCK_M:-}"   # TILE
TILE_BLOCK_N="${MOE_DRIVER_BLOCK_N:-}"   # TILE
TILE_NUM_WARPS="${MOE_DRIVER_NUM_WARPS:-}"   # TILE
CENSUS_TILES="${MOE_DRIVER_CENSUS_TILES:-}"   # TILE
CENSUS_MODELS="${MOE_DRIVER_CENSUS_MODELS:-}"   # TILE
#: A2: a gate unit's gate, a lockprobe unit's ladder and probe pages, a timed unit's own lock   # A2
GATE_NAME="${MOE_DRIVER_GATE:-}"   # A2
LADDER="${MOE_DRIVER_LADDER:-}"   # A2
PROBES="${MOE_DRIVER_PROBES:-}"   # A2
UNLESS_LOCK="${MOE_DRIVER_UNLESS_LOCK:-}"   # A2
#: S1: a bytes unit's ladder treads, its replay mode and its stall metrics; the prelude's power read-back   # S1
BYTE_TREADS="${MOE_DRIVER_BYTE_TREADS:-}"   # S1
S1_REPLAY="${MOE_DRIVER_S1_REPLAY:-}"   # S1
S1_STALL="${MOE_DRIVER_S1_STALL:-}"   # S1
PL_WAIT_S="${MOE_DRIVER_PL_WAIT_S:-30}"   # S1
PL_POLL_S="${MOE_DRIVER_PL_POLL_S:-2}"   # S1
POWERPROOF_S=90   # S1: S1-M0's GEMM seconds (the registration's)
S1_FALLBACK_W=700   # S1: the owner's pre-authorised fallback limit (2026-10-21)
INSTR_VARIANTS="${MOE_DRIVER_INSTR_VARIANTS:-}"
VARIANT_IDS="${MOE_DRIVER_VARIANT_IDS:-}"
#: the registered stamp variants (moe/instrumented R5_VARIANTS; a test holds these to it):
#: v1 the sample level on every CTA, v2 ends on every live CTA with iteration tops and dead
#: CTAs 1 in 17, v3 every stamp on 1 in 17 CTAs (pid mod 17, never 16: design-r5-review e2)
r5_variant_spec() {
  case "$1" in
    v1) echo "stamps=sample,every=16,cta_mod=1,iter_mod=1,dead_mod=1" ;;
    v2) echo "stamps=sample,every=16,cta_mod=1,iter_mod=17,dead_mod=17" ;;
    v3) echo "stamps=sample,every=16,cta_mod=17,iter_mod=17,dead_mod=17" ;;
  esac
}
#: the deepest tread a histogram page asks (its dry run's "retracted tread"), for check_dry
hist_deepest() {
  python3 -c 'import json,sys; print(max(int(c["n"]) for c in json.load(open(sys.argv[1]))["cells"]))' "$1" 2>/dev/null
}
#: gpu-benches lives here on the VM, outside the checkout (GPL-3.0, never in the repo)
GPUBENCH_DEST="${MOE_DRIVER_GPUBENCH_DEST:-$MOE_HOME/ext/gpu-benches}"
GPUBENCH_L2_MIB=60
#: RENTAL 2's knobs (2026-10-01; docs/registered/README.md, rental 2). A bytes
#: unit's config values for the existing kernel and its partition metrics; a floor
#: unit's lock (a lock other than 1710 is the one the prelude probed), whether its
#: base-clock captures run, its shape metrics and its null kernel. Empty: as before.
NUM_STAGES="${MOE_DRIVER_NUM_STAGES:-}"
BLOCK_K="${MOE_DRIVER_BLOCK_K:-}"
SLOT_PAD_ROWS="${MOE_DRIVER_SLOT_PAD_ROWS:-}"
PARTITION_METRICS="${MOE_DRIVER_PARTITION_METRICS:-}"
FLOOR_LOCK_MHZ="${MOE_DRIVER_FLOOR_LOCK_MHZ:-}"
FLOOR_BASE="${MOE_DRIVER_FLOOR_BASE:-1}"
FLOOR_SHAPE="${MOE_DRIVER_FLOOR_SHAPE:-}"
FLOOR_NULL="${MOE_DRIVER_FLOOR_NULL:-}"
#: Locks besides 1710 that a plan's floor units name: the prelude locks each one
#: once and records whether it held (locks.env LOCK_PROBE_<F>), so a lock this card
#: refuses costs no unit's minutes (rental 2's review, section 6).
PROBE_LOCKS="${MOE_DRIVER_PROBE_LOCKS:-}"
#: launchfloor's design (scripts/launch_floor.py; docs/registered 2026-10-01).
LF_TREADS="${MOE_DRIVER_LF_TREADS:-1,2,3,4,5,6,7,8,9}"
LF_MODES="${MOE_DRIVER_LF_MODES:-E240,E0,E480,GR}"
LF_TRACE="${MOE_DRIVER_LF_TRACE:-1,2}"
#: RENTAL 3 (2026-10-05): the launch floor's arms and G (lf-arms, lf-group-m); lf-trace=0
#: runs no trace phase (the G = 64 NATIVE wall-time check times only).
LF_ARMS="${MOE_DRIVER_LF_ARMS:-native,shared,private}"
LF_GROUP_M="${MOE_DRIVER_LF_GROUP_M:-4}"
R3_BASE=(--model "$MODEL" --block-m "${TILE_BLOCK_M:-32}" --repeats 9 --duty 0.25 --seed 0)   # TILE
#: R1 in lock mode (section 3b); each state a held SM clock.
R1_BASE=(--model "$MODEL" --dtype bf16 --treads 8 --repeats 13 --burst-ms 40
         --target-ms 200 --trials 3 --warm-ms 200 --settle-seconds 10 --lock-duty 0.25)
#: R1's claim spans down to this clock: its lowest lock must be at or under it.
R1_SPAN_MAX_LOW=1459
#: The GH200's 2026-09-25 R3 pages timed at 1710 (session tag ending
#: -lock1710), which C5 scores the byte pages against.
TIMED_REF_DIR="results/published/2026-09-25-nvidia_gh200_480gb-session/results/gaps-nvidia_gh200_480gb/private_weight_reference"
TIMED_REF_IDS=(d9f1f37c df37ea07 01c08abd 1b285de2 6ff34777)
#: The floor's G=2 capture is dropped when the G=64 capture's ncu log shows
#: more replay passes than this per launch (the draft's rule).
FLOOR_MAX_PASSES=50
#: The NATIVE treads each floor capture profiles, for the plan to print. The
#: capture itself takes them from dram_counter_route.r3_floor_treads by --model
#: (2 3 4 6 unless a GEMM has fewer than 3 cells of at least 4 waves); this
#: table is what that function returns, and a test pins it for every model.
floor_treads() {
  case "$MODEL" in
    jetmoe-8b) echo "2 3 4 6 9 10 11" ;;
    mixtral-8x7b|mixtral-8x7b-tp2|mixtral-8x7b-tp4|mixtral-8x7b-tp8|granite-3.0-3b-a800m|granite-3.0-1b-a400m|qwen2-57b-a14b-tp8) echo "2 3 4 6 7 8" ;;
    *) echo "2 3 4 6" ;;
  esac
}

# ---- the steps: name, estimate and cap in minutes, what it answers ----------
STEPS=(prelude bytes calibrate timed eta floor deep r1lock launchfloor rulers gpubench regcheck perturb stamps tilecensus lockprobe gate powerproof)   # TILE A2 S1
#: Steps no default plan runs: asked for by --steps or a --plan line only.
OPT_IN_STEPS=(launchfloor rulers gpubench regcheck perturb stamps tilecensus lockprobe gate powerproof)   # TILE A2 S1
#: Another model's pages run longer by about its weight bytes over 8x7B's
#: (8x22B: 4.83 GB against 2.82, x1.7); the estimates and caps below are
#: 8x7B's, scaled by this percentage for the steps that measure.
#: Qwen2-57B-A14B: 3.52 GB, x1.25. OLMoE-1B-7B 0.81 GB, Qwen1.5-MoE 1.04, Phi-3.5-MoE 2.52, JetMoE-8B 0.55, Granite-3.0-3B 0.19: held at x1 (their pages run as fast as 8x7B's or faster; the estimates stay conservative).
model_scale_pct() {
  case "$MODEL" in "$DEFAULT_MODEL") echo 100 ;; mixtral-8x22b) echo 170 ;; qwen2-57b-a14b) echo 125 ;; olmoe-1b-7b|qwen1.5-moe-a2.7b|phi-3.5-moe|jetmoe-8b|granite-3.0-3b-a800m) echo 100 ;; *) echo 200 ;; esac
}
_scaled() { case "$1" in prelude|calibrate|rulers|gpubench|regcheck|perturb|tilecensus|lockprobe|gate|powerproof) echo "$2" ;; *) echo $(( $2 * $(model_scale_pct) / 100 )) ;; esac; }   # TILE A2 S1
step_est() {
  local m
  case "$1" in
    prelude) m=3 ;; bytes) m=55 ;; calibrate) m=8 ;; timed) m=37 ;;
    eta) m=50 ;; floor) m=15 ;; deep) m=33 ;; r1lock) m=62 ;; launchfloor) m=25 ;;
    rulers) m=3 ;; gpubench) m=8 ;; regcheck) m=3 ;; perturb) m=10 ;; stamps) m=3 ;;
    tilecensus) m=8 ;;   # TILE
    lockprobe) m=16 ;; gate) m=1 ;;   # A2
    powerproof) m=3 ;;   # S1
  esac
  _scaled "$1" "$m"
}
step_cap() {
  local m
  case "$1" in
    prelude) m=15 ;; bytes) m=120 ;; calibrate) m=30 ;; timed) m=90 ;;
    eta) m=115 ;; floor) m=45 ;; deep) m=75 ;; r1lock) m=130 ;; launchfloor) m=60 ;;
    rulers) m=10 ;; gpubench) m=20 ;; regcheck) m=10 ;; perturb) m=25 ;; stamps) m=10 ;;
    tilecensus) m=25 ;;   # TILE
    lockprobe) m=50 ;; gate) m=5 ;;   # A2
    powerproof) m=10 ;;   # S1
  esac
  _scaled "$1" "$m"
}
#: THE TILE STAGE: R3's script, the counter route's, and their tile arguments (nothing at R3's tile)   # TILE
r3_script() { if [[ -n "$TILE_NUM_WARPS" ]]; then echo scripts/r3_tile.py; else echo scripts/private_weight_reference.py; fi; }   # TILE
dcr_script() { if [[ -n "$S1_REPLAY$S1_STALL" ]]; then echo scripts/dram_counter_route_s1.py; elif [[ -n "$TILE_NUM_WARPS" ]]; then echo scripts/dram_counter_route_tile.py; else echo scripts/dram_counter_route.py; fi; }   # TILE S1
#: S1: the S1 wrapper's own flags (scripts/dram_counter_route_s1.py), nothing on any other unit   # S1
s1_dcr_args() {   # S1
  [[ -n "$S1_REPLAY" ]] && printf '%s\n' --s1-replay "$S1_REPLAY"   # S1
  [[ "$S1_STALL" == 1 ]] && printf '%s\n' --s1-stall-metrics   # S1
  return 0   # S1
}   # S1
s1_text() { local t=""; [[ -n "$BYTE_TREADS" ]] && t+="; ladder treads ${BYTE_TREADS//,/ }"; [[ -n "$S1_REPLAY" ]] && t+="; ncu ${S1_REPLAY} replay, no cache flush"; [[ "$S1_STALL" == 1 ]] && t+="; stall metrics"; echo "$t"; }   # S1
#: S1 (the A2 defect): after -pl, poll until the ENFORCED GPU limit (--query-gpu=enforced.power.limit, the minimum of   # S1
#: every limiter) and -q -d POWER's GPU-scope Current Power Limit both read the plan's limit within 1 W; only then   # S1
#: write power-set.txt. A2 read power.limit (the software request, 900 W) and -q (700 W) disagree all session. Never: 1.   # S1
s1_power_settle() {   # S1
  local waited=0 qpl qreq epl cpl q   # S1
  while :; do   # S1
    q="$(nvidia-smi -q -d POWER 2>&1)"   # S1
    qpl="$(awk '/GPU Power Readings/ { g = 1 } g && /Current Power Limit/ { print $5; exit }' <<< "$q")"   # S1
    qreq="$(awk '/GPU Power Readings/ { g = 1 } g && /Requested Power Limit/ { print $5; exit }' <<< "$q")"   # S1
    epl="$(nvidia-smi --query-gpu=enforced.power.limit --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"   # S1
    cpl="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"   # S1
    if awk -v a="$qpl" -v b="$epl" -v w="$POWER_LIMIT_W" 'BEGIN { exit !(a ~ /^[0-9.]+$/ && b ~ /^[0-9.]+$/ && (a - w) ^ 2 <= 1 && (b - w) ^ 2 <= 1) }'; then   # S1
      printf '%s\n' "$q" > "$S/power-set.txt"   # S1
      ledger "prelude: the power limit read back after ${waited} s: enforced.power.limit ${epl} W, -q -d POWER GPU scope Current ${qpl} W (Requested ${qreq} W), power.limit ${cpl} W (S1 read-back)"   # S1
      return 0   # S1
    fi   # S1
    if (( waited >= PL_WAIT_S )); then   # S1
      printf '%s\n' "$q" > "$S/power-set-unsettled-${POWER_LIMIT_W}.txt"   # S1 C2: the file is named by its limit
      ledger "prelude: after ${waited} s enforced.power.limit reads '${epl}' W and -q -d POWER GPU scope Current '${qpl}' W (Requested '${qreq}' W; power.limit '${cpl}' W), not both $POWER_LIMIT_W W (power-set-unsettled-${POWER_LIMIT_W}.txt)"   # S1 C2
      return 1   # S1
    fi   # S1
    sleep "$PL_POLL_S"; waited=$(( waited + PL_POLL_S ))   # S1
  done   # S1
}   # S1
#: S1: the lock a unit runs at, for the ledger's START line (A2's printed the default 1710: the orchestrator never   # S1
#: read a2-lock.env): the unit's own lock-mhz, else the session's (a2-lock.env), else the plan prelude's, else 1710   # S1
s1_unit_lock() {   # S1
  local l="${MOE_DRIVER_UNIT_LOCK:-}"   # S1
  [[ -z "$l" && -n "${D:-}" ]] && l="$(sed -n 's/^A2_LOCK=//p' "$D/a2-lock.env" 2>/dev/null)"   # S1
  [[ -z "$l" ]] && l="$(grep -oE 'lock-mhz=[0-9]+' <<< "${U_OPTS[0]:-}" | cut -d= -f2)"   # S1
  echo "${l:-$LOCK_TIMED}"   # S1
}   # S1
s1_what() { local LOCK_TIMED; LOCK_TIMED="$(s1_unit_lock)"; step_what "$1"; }   # S1
#: C2: the limit a unit runs at, for the dry run and the ledger's START line (S1's printed the default 900 W: the   # C2
#: orchestrator exports the prelude's power-limit-w to the prelude but never set its own): the plan prelude's   # C2
c2_plan_power() { local w; w="$(grep -oE 'power-limit-w=[0-9]+' <<< "${U_OPTS[0]:-}" | cut -d= -f2)"; echo "${w:-$POWER_LIMIT_W}"; }   # C2
c2_what() { local POWER_LIMIT_W; POWER_LIMIT_W="$(c2_plan_power)"; s1_what "$1"; }   # C2
lr3_script() { [[ -n "$TILE_NUM_WARPS" ]] && printf '%s\n' --r3-script scripts/r3_tile.py; return 0; }   # TILE
tile_r3_args() {   # TILE
  [[ -n "$TILE_BLOCK_N" ]] && printf '%s\n' --block-n "$TILE_BLOCK_N"   # TILE
  [[ -n "$TILE_NUM_WARPS" ]] && printf '%s\n' --num-warps "$TILE_NUM_WARPS"   # TILE
  return 0   # TILE
}   # TILE
tile_dcr_args() {   # TILE
  [[ -n "$TILE_NUM_WARPS" ]] && printf '%s\n' --num-warps "$TILE_NUM_WARPS"   # TILE
  [[ -n "$TILE_BLOCK_M" ]] && printf '%s\n' --block-m "$TILE_BLOCK_M"   # TILE
  [[ -n "$TILE_BLOCK_N" ]] && printf '%s\n' --block-n "$TILE_BLOCK_N"   # TILE
  return 0   # TILE
}   # TILE
tile_text() { [[ -n "$TILE_BLOCK_M$TILE_BLOCK_N$TILE_NUM_WARPS" ]] && echo "; tile ${TILE_BLOCK_M}x${TILE_BLOCK_N}x${BLOCK_K:-64} w${TILE_NUM_WARPS} s${NUM_STAGES:-4}"; return 0; }   # TILE
#: locked_r3.py's cap on one R3 run (its default, 1800 s, fits 8x7B's pages);
#: another model's deep page runs about 27 min, so it takes an hour.
lr3_cap() { [[ "$MODEL" == "$DEFAULT_MODEL" ]] || printf '%s\n' --run-cap-s 3600; }
#: rental 6: a timed unit's slip-policy=cell, handed to locked_r3 (nothing under page)
lr3_slip() { [[ "$SLIP_POLICY" == cell ]] && printf '%s\n' --slip-policy cell; return 0; }
#: rental 7: under g1-rule=r7-median-run3 locked_r3 takes --min-clean-repeats 0 (R3 keeps every repeat)
lr3_minclean() { [[ "$G1_RULE" == r7-median-run3 ]] && printf '%s\n' --min-clean-repeats 0; return 0; }
#: rental 7: the slip arguments locked_r3 --slip-policy cell --min-clean-repeats 0 hands R3, for the
#: unit's R3 dry run (so the dry run checks what will run); nothing otherwise
dry_slip() {
  [[ "$G1_RULE" == r7-median-run3 ]] && printf '%s\n' --slip-policy cell --lock-mhz "$LOCK_TIMED" --min-clean-repeats 0
  return 0
}
#: Set an nvidia-smi lock and confirm it by reading it back, once more if the
#: first read-back is off (2026-09-28: the 8x22B floor's lock capture ran at
#: 1800 MHz, its own nvidia-smi records reading 1980 before and after, though
#: `-lgc 1710` had printed "GPU clocks set"). Idle, a held lock reads F.
lock_and_check() {   # F
  local f=$1 got try
  for try in 1 2; do
    sudo -n nvidia-smi -lgc "$f,$f" >/dev/null || return 2
    sleep 2
    got=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [ "$got" = "$f" ] && return 0
    echo "lock $f MHz not in force after -lgc (clocks.sm reads $got); try $try"
  done
  return 2
}
#: The census the byte and floor pages read: the preflight's (PF6, 8x7B's) for
#: the study's model, else one taken for this model at ncu's base clock.
census_for() {
  if [[ "$MODEL" == "$DEFAULT_MODEL" ]]; then printf '%s\n' "$S/census.json"; return 0; fi
  local C="$S/census-$MODEL.json"
  if [[ ! -s "$C" ]]; then
    moe_counter "$PY_VLLM" scripts/dram_counter_route.py --run --family r3-arms --census-only \
      --model "$MODEL" --out "$C" > "$S/logs/census-$(uid).log" 2>&1 \
      || { ledger "census for $MODEL failed (logs/census-$(uid).log)" >&2; return 1; }
    moe_counter ncu --clock-control reset >/dev/null 2>&1 || true
    ledger "census for $MODEL written ($C)" >&2
  fi
  printf '%s\n' "$C"
}
step_what() {
  case "$1" in
    prelude)   echo "3c.1 supported clocks, persistence, ${POWER_LIMIT_W} W (power.max_limit read first), the session lock ${MOE_DRIVER_A2_LOCK:-$LOCK_TIMED} MHz" ;;   # A2
    bytes)     if [[ -n "$BYTE_GROUPS" ]]; then echo "3c.2 byte pages at the ${LOCK_TIMED} lock, the registered G = ${BYTE_GROUPS//,/ }, treads ${BYTE_TREADS:-$TREADS}; no base-clock control$(knob_text)$(tile_text)$(s1_text)"   # TILE S1
               else echo "3c.2 byte pages at the ${LOCK_TIMED} lock, G = ${BYTE_GS[*]}, treads $TREADS; base-clock control$(knob_text)"; fi ;;
    calibrate) echo "3c.3 calibrate (the ruler), no lock in force" ;;
    timed)     if [[ -n "$HISTOGRAM" ]]; then echo "3c.3 timed R3 histogram page at ${MOE_DRIVER_UNIT_LOCK:-$LOCK_TIMED}$(: A2): G=$(timed_groups | tr ' ' ','), $HISTOGRAM, arms ${R3_ARMS:-native,shared}, ${DECLARED_COPIES:-?} copies$(timed_knob_text)$([[ "$SLIP_POLICY" == cell ]] && echo ', slip-policy cell')$([[ -n "$G1_RULE" ]] && echo ", g1-rule $G1_RULE")$(tile_text)"   # TILE
               elif [[ -n "$TIMED_GROUPS$TIMED_TREADS" ]]; then echo "3c.3 timed R3 at ${LOCK_TIMED}: G=$(timed_groups | tr ' ' ',') (treads $(timed_treads)), no G=3 page"
               else echo "3c.3 timed R3 at ${LOCK_TIMED}: G=8, 32 (treads 6), G=3 (treads 8): P2, P5"; fi ;;
    eta)       echo "3c.4 timed R3 at held locks 1410 (G=4, 2), 1500 and 1605 (G=4): P1" ;;
    floor)     if [[ -n "$FLOOR_GROUPS" ]]; then echo "3c.5 registered floor design: NATIVE at G = ${FLOOR_GROUPS//,/ }, treads ${FLOOR_TREADS:-default}, $([[ "$FLOOR_BASE" == 0 ]] && echo "no base-clock capture" || echo "base clock") and the ${FLOOR_LOCK_MHZ:-$LOCK_TIMED} lock$(floor_knob_text)"
               else echo "3c.5 floor counters, G=64 and G=2 at base, G=64 unlocked (record), G=64 at ${LOCK_TIMED}: P6"; fi ;;
    deep)      echo "3c.6 timed R3 at ${LOCK_TIMED} to tread 9, G=4 then G=2: P3, P4" ;;
    r1lock)    echo "3c.7 R1 in lock mode, G=4 then G=1: the floor's clock exponent" ;;
    rulers)    echo "rulers at the ${MOE_DRIVER_UNIT_LOCK:-$LOCK_TIMED}$(: A2) lock: torch read2d / copy / add / matmul bf16 8192^3, the device's L2 size" ;;
    gpubench)  echo "THIRD-PARTY RRZE-HPC/gpu-benches (GPL-3.0) at the ${LOCK_TIMED} lock: latency, stream, l2-cache; fetched into \$MOE_HOME/ext, never the repo" ;;
    regcheck)  echo "regcheck, compile only (no lock, nothing timed): every instrumented variant's CTAs per SM against the plain kernel's (rental 5)" ;;
    perturb)   echo "perturbation gate at the ${LOCK_TIMED} lock: plain vs the instrumented copy on every later instrumented unit's cells (decision 2)" ;;
    stamps)    [[ -n "$HISTOGRAM" ]] && { echo "stamps at the ${LOCK_TIMED} lock: the instrumented copy ($(stamp_spec)) at ${TILE_BLOCK_M}x${TILE_BLOCK_N}x${BLOCK_K} w${TILE_NUM_WARPS} s${NUM_STAGES} G=${STAMP_GROUPS}, $HISTOGRAM, arms ${STAMP_ARMS//,/ }; runs on a PASS regcheck and gate hard legs, PERTURBED on a timing FAIL (scripts/instr_probe_c3.py)"; return 0; }   # C3
               echo "stamps at the ${LOCK_TIMED} lock: the instrumented copy ($([[ -n "$INSTR_VARIANTS" ]] && echo "the first passing of ${INSTR_VARIANTS//,/ }" || stamp_spec)), G=${STAMP_GROUPS//,/ } treads ${STAMP_TREADS//,/ } arms ${STAMP_ARMS//,/ }, BK ${BLOCK_K:-64} s${NUM_STAGES:-4}; runs only on a PASS gate" ;;   # C3 (C2's line, under the histogram branch)
    tilecensus) echo "tile census, compile only (no lock, nothing timed): tiles ${CENSUS_TILES//,/ } on ${CENSUS_MODELS//,/ }; refused on an MMA path mismatch" ;;   # TILE
    lockprobe) echo "the lock-hold probe: per rung of ${LADDER//,/ } MHz, hw_rulers then the probe pages at the rung, the gate; the first rung that holds is the lock, none: STOP" ;;   # A2
    powerproof) echo "the power proof: a sustained bf16 GEMM for ${POWERPROOF_S} s with no clock lock, nvidia-smi logging power.draw, enforced.power.limit and clocks_event_reasons every 100 ms (S1-M0)" ;;   # S1
    gate)      echo "gate ${GATE_NAME}: scripts/tileA2_gate.py ${GATE_NAME} at the ${LOCK_TIMED} lock; FAIL ends the rental (tileA2-gates)" ;;   # A2
    launchfloor) if [[ "$LF_TRACE" == 0 ]]; then echo "launch floor at the ${LOCK_TIMED} lock: arms $LF_ARMS, G=$LF_GROUP_M, treads $LF_TREADS, modes $LF_MODES; a timed process (no profiler), no trace phase"
                 else echo "launch floor at the ${LOCK_TIMED} lock: arms $LF_ARMS, G=$LF_GROUP_M, treads $LF_TREADS, modes $LF_MODES, traces at $LF_TRACE; a timed process (no profiler) then a trace process"; fi ;;
  esac
}
#: Rental 2's bytes knobs as dram_counter_route.py's flags, and as text.
knob_args() {
  [[ -n "$NUM_STAGES" ]] && printf '%s\n' --num-stages "$NUM_STAGES"
  [[ -n "$BLOCK_K" ]] && printf '%s\n' --block-k "$BLOCK_K"
  [[ -n "$SLOT_PAD_ROWS" ]] && printf '%s\n' --slot-pad-rows "$SLOT_PAD_ROWS"
  [[ "$PARTITION_METRICS" == 1 ]] && printf '%s\n' --partition-metrics
  [[ -n "$INSTR_EVICT_A$INSTR_EVICT_B" ]] && printf '%s\n' --instr "$(evict_spec)"
  [[ -n "$HISTOGRAM" ]] && printf '%s\n' --histogram "$HISTOGRAM" --arms "${R3_ARMS:-native,shared}"
  return 0
}
#: rental 5: a timed unit's config values for R3 (block-k, num-stages), as flags and as text
timed_knob_args() {
  [[ -n "$NUM_STAGES" ]] && printf '%s\n' --num-stages "$NUM_STAGES"
  [[ -n "$BLOCK_K" ]] && printf '%s\n' --block-k "$BLOCK_K"
  return 0
}
timed_knob_text() {
  local t; t="$(timed_knob_args | tr '\n' ' ')"
  [[ -n "$t" ]] && echo "; knobs ${t% }"
  return 0
}
#: rental 4: a bytes unit's eviction-hint spec, and a stamps unit's stamp spec (moe.instrumented)
evict_spec() { echo "evict_a=${INSTR_EVICT_A:-none},evict_b=${INSTR_EVICT_B:-none}"; }
stamp_spec() {
  local t="stamps=${INSTR_STAMPS:-off}"
  [[ "$INSTR_STAMPS" == iter ]] && t+=",every=${INSTR_EVERY:-1}${INSTR_MARKS:+,marks=$INSTR_MARKS}"
  echo "$t"
}
#: the declaration a timed unit's R3 dry run must print (check_dry)
decl_expect() { echo "${DECLARED_COPIES:-$COUNTER_COPIES}"; }
knob_text() {
  local t; t="$(knob_args | tr '\n' ' ')"
  [[ -n "$t" ]] && echo "; knobs ${t% }"
  return 0
}
#: Rental 2's floor knobs as flags, and as text.
floor_knob_args() {
  [[ "$FLOOR_SHAPE" == 1 ]] && printf '%s\n' --floor-shape-metrics
  [[ "$FLOOR_NULL" == 1 ]] && printf '%s\n' --floor-null-kernel
  return 0
}
floor_knob_text() {
  local t; t="$(floor_knob_args | tr '\n' ' ')"
  [[ -n "$t" ]] && echo "; ${t% }"
  return 0
}
#: A timed unit's groups and treads (rental 3's timed-groups / timed-treads).
timed_groups() { if [[ -n "$TIMED_GROUPS" ]]; then echo "${TIMED_GROUPS//,/ }"; else echo "8 32"; fi; }
timed_treads() { echo "${TIMED_TREADS:-6}"; }
#: R3's results directory for a timed step: the session's one directory, or a plan
#: unit's own (rental 3, 2026-10-05: gaps-<card>-<model>-<label>).
results_dir() { echo "$RESULTS_ROOT/gaps-$MOE_CARD${OUT_LABEL:+-$MODEL-$OUT_LABEL}"; }
#: What the deadline drops, in the draft's order, with the minutes each saves.
DROPS=(r1_g1 r1_g4 eta_1605 bytes_tail)
drop_step() { case "$1" in r1_g1|r1_g4) echo r1lock ;; eta_1605) echo eta ;; bytes_tail) echo bytes ;; esac; }
drop_min()  { case "$1" in r1_g1|r1_g4) echo 31 ;; eta_1605) echo 12 ;; bytes_tail) echo 14 ;; esac; }
drop_what() {
  case "$1" in
    r1_g1) echo "R1 at G=1" ;; r1_g4) echo "R1 at G=4" ;; eta_1605) echo "the 1605 lock" ;;
    bytes_tail) echo "the G=64 byte page and the base-clock control" ;;
  esac
}
RESERVE_MIN="${MOE_DRIVER_RESERVE_MIN:-8}"
BUSY_WAIT_S="${MOE_DRIVER_BUSY_WAIT_S:-300}"

# ---- small tools ---------------------------------------------------------
utc()    { date -u +%Y-%m-%dT%H:%M:%SZ; }
now_s()  { if [[ -n "${MOE_DRIVER_NOW:-}" ]]; then echo "$MOE_DRIVER_NOW"; else date +%s; fi; }
utc_of() { date -u -d "@$1" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -r "$1" +%Y-%m-%dT%H:%M:%SZ; }
sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}
#: One UTC line in the driver's ledger, and on the console.
ledger() {
  local line; line="$(utc) $*"
  printf '%s\n' "$line"
  [[ -n "${D:-}" ]] && { mkdir -p "$D"; printf '%s\n' "$line" >> "$D/status"; }
  return 0
}
dropped() { [[ -f "$D/drops" ]] && grep -qx -- "$1" "$D/drops"; }
#: RENTAL 7 (design-r7 S9, design-r7-review (f)): every per-unit log under $S/logs is named by
#: the unit's model and label (the model alone when a step runs with no label), so no two units
#: share one. On rental 6 the dry runs of OLMoE and Phi at deepest tread 12 (and of Phi and
#: Qwen1.5 at t2 and t4) wrote one file, both byte units wrote r3c-g8-lock1710.log (Phi's alone
#: survived), and both second-K units shared their logs. What the driver itself reads back
#: (calibrate's RESULT lines, the floor's r3f-g64 replay passes, the byte page's VALIDITY lines,
#: a dry run's checks) reads the same per-unit name it wrote.
uid() { printf '%s' "$MODEL${OUT_LABEL:+-$OUT_LABEL}"; }
#: A session tag no earlier run of this driver used: BASE, else BASEb, BASEc...
#: locked_r3.py refuses a tag whose ledger or pages exist, so a rerun of a step
#: takes the next letter rather than being refused.
#: After the ten letter suffixes ("", b to j, the names every session through
#: rental 6 used) it counts on, BASE-11, BASE-12, ..., to 100000: rental
#: 6's eleventh timed unit found the ten taken, got an empty tag, and its dry run
#: read `--session-tag -lock1710`, which argparse refused, as it did for every
#: timed unit after it (units 13 to 15, 18 and 19). Every caller checks the
#: return, so a tag can never be empty.
fresh_tag() {
  local base="$1" s t i=0 letters=("" b c d e f g h i j)
  while (( i < 100000 )); do
    if (( i < ${#letters[@]} )); then s="${letters[i]}"; else s="-$(( i + 1 ))"; fi
    i=$(( i + 1 ))
    t="$base$s"
    if ! grep -qx -- "$t" "$D/tags-used" 2>/dev/null && [[ ! -e "$SESSION_ROOT/locked-r3/$t" ]]; then
      printf '%s\n' "$t" >> "$D/tags-used"
      printf '%s\n' "$t"
      return 0
    fi
  done
  return 1
}
gpu_busy() { nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>&1; }
#: Every ledger line of a locked_r3.py run that names no card.
ledger_lines() { grep -hE 'G=[0-9]+ [a-z_-]+ at lock [0-9]+: R3 exit' "$@" 2>/dev/null || true; }

#: R3's own dry run may go on to a lock: no REFUSED line, the 9-copy
#: declaration, a retracted tread of at least the treads planned, and a ruler
#: (not the H200 HYPOTHESIS R3 falls back to without one, under which it
#: refuses the timed run).
check_dry() {
  local log="$1" planned="$2" n
  if grep -q '^REFUSED:' "$log"; then
    echo "REFUSED: $(grep -m1 '^REFUSED:' "$log" | cut -c10-160)"; return 1
  fi
  local want; want="$(decl_expect)"
  grep -qE "n_decl = $want against n_max" "$log" || { echo "no 'n_decl = $want against n_max' line"; return 1; }
  n="$(grep -oE 'retracted +tread +[0-9]+ +against the [0-9]+ planned' "$log" | head -1 | awk '{print $3}')"
  [[ -n "$n" && "$n" -ge "$planned" ]] || { echo "retracted tread ${n:-(none printed)} is under the $planned planned"; return 1; }
  if grep -qE '^ *ridge .*HYPOTHESIS' "$log"; then
    echo "no ruler: the ridge is R3's H200 HYPOTHESIS"; return 1
  fi
  return 0
}

#: The ruler arguments every timed step hands R3: none when calibrate's ruler
#: stands, the 2026-09-25 numbers when it failed or never ran.
load_ruler() {
  RULER_ARGS=()
  RULER_KIND=""
  if [[ -f "$D/ruler.env" ]]; then
    # shellcheck disable=SC1090
    . "$D/ruler.env"
  elif [[ -f "$REPO/moe/bench/hardware/measured_$MOE_CARD.yaml" ]]; then
    RULER_KIND=measured
  else
    RULER_KIND=fallback
    RULER_ARGS=("${RULER_FALLBACK[@]}")
    ledger "no ruler decision on record (calibrate did not run): R3 takes the 2026-09-25 ruler's numbers, ${RULER_FALLBACK[*]}"
  fi
}

#: The locks the prelude resolved from this card's supported list.
load_locks() {
  LOCK_ETA_A=""; LOCK_ETA_B=""; LOCK_ETA_C=""
  # shellcheck disable=SC1090
  [[ -f "$D/locks.env" ]] && . "$D/locks.env"
  : "${LOCK_ETA_A:=${P1_LOCKS[0]}}" "${LOCK_ETA_B:=${P1_LOCKS[1]}}" "${LOCK_ETA_C:=${P1_LOCKS[2]}}"
}

#: The step-mode prologue: the VM's environment and the checkout.
enter() {
  [[ -f "$ENV_SH" ]] || { echo "no $ENV_SH: run setup_vm.sh first (docs/LAMBDA.md section 2)"; exit "$EXIT_REFUSED"; }
  # shellcheck disable=SC1090
  . "$ENV_SH"
  cd "$REPO" || exit "$EXIT_REFUSED"
  S="$SESSION_ROOT"
  D="$S/gh200-driver"
  mkdir -p "$S/logs" "$D"
  B="$(cat "$S/base-tag.txt" 2>/dev/null || true)"
  # Every later step needs what the prelude set: the base tag, the lock plan,
  # persistence and the 700 W limit.
  if [[ "$1" != prelude && ( -z "$B" || ! -f "$D/locks.env" ) ]]; then
    echo "step $1 REFUSED: the prelude has not passed on this VM (no base-tag.txt or locks.env)"
    exit "$EXIT_REFUSED"
  fi
  # A2: the session's lock (the prelude's lock-mhz, then the lock-hold probe's rung), then a timed unit's own
  A2_LOCK=""   # A2
  [[ -f "$D/a2-lock.env" ]] && . "$D/a2-lock.env"   # A2
  if [[ "$1" == prelude ]]; then LOCK_TIMED="${MOE_DRIVER_A2_LOCK:-$LOCK_TIMED}"; else LOCK_TIMED="${A2_LOCK:-$LOCK_TIMED}"; fi   # A2
  LOCK_TIMED="${MOE_DRIVER_UNIT_LOCK:-$LOCK_TIMED}"   # A2
}

# ==========================================================================
# 3c.1 prelude
# ==========================================================================
step_prelude() {
  [[ -s "$S/base-tag.txt" ]] || echo "gh200-$(date -u +%Y%m%dT%H%M%SZ)" > "$S/base-tag.txt"
  B="$(cat "$S/base-tag.txt")"
  echo "base tag $B (every locked run adds its own suffix)"
  git rev-parse HEAD | tee "$S/commit.txt"
  git status --porcelain | tee "$S/git-at-start.txt"
  if [[ -s "$S/git-at-start.txt" ]]; then
    ledger "prelude REFUSED: the checkout is not clean (git-at-start.txt)"; return "$EXIT_REFUSED"
  fi
  if ! sudo -n true 2>/dev/null; then
    ledger "prelude REFUSED: no passwordless sudo, which every lock needs"; return "$EXIT_REFUSED"
  fi
  local busy; busy="$(gpu_busy)"
  if [[ -n "$busy" ]]; then
    ledger "prelude REFUSED: the GPU is in use: $busy"; return "$EXIT_REFUSED"
  fi
  if ! nvidia-smi -q -d SUPPORTED_CLOCKS > "$S/supported-clocks.txt"; then
    ledger "prelude REFUSED: nvidia-smi could not list the supported clocks"; return "$EXIT_REFUSED"
  fi
  # Which of the registered locks this card supports, and the locks the later
  # steps take: 1710 or nothing; 1605, else its registered 1590; 1410 and 1500,
  # else the nearest supported clock one 15 MHz step away, labelled.
  if ! python3 - "$S/supported-clocks.txt" "$LOCK_TIMED" "${P1_LOCKS[@]}" "$P1_FALLBACK_1605" \
      ${PROBE_LOCKS//,/ } > "$S/lock-plan.txt" <<'PY'
import re, sys
g = sorted({int(x) for x in re.findall(r"Graphics\s*:\s*(\d+)\s*MHz", open(sys.argv[1]).read())})
if not g:
    sys.exit("no graphics clock read: look at supported-clocks.txt")
timed, a, b, c, c_fb = map(int, sys.argv[2:7])
probe = [int(x) for x in sys.argv[7:]]
print(f"# {len(g)} graphics clocks, {g[0]} to {g[-1]} MHz")
for f in (timed, c, c_fb, b, a):
    print(f"# {f} " + ("supported" if f in g else
                       f"NOT supported; nearest {[s for s in g if abs(s - f) <= 15]}"))
def near(f):
    c = [s for s in g if abs(s - f) <= 15]
    return min(c, key=lambda s: (abs(s - f), -s)) if c else None
out = {"LOCK_TIMED_OK": 1 if timed in g else 0}
for name, f in (("LOCK_ETA_A", a), ("LOCK_ETA_B", b)):
    if f in g:
        out[name], note = f, "registered"
    elif near(f):
        out[name], note = near(f), f"SUBSTITUTED for {f}: P1 at it is computed after registration"
    else:
        out[name], note = "none", f"{f} and every clock within 15 MHz unsupported: dropped"
    print(f"# {name} {out[name]} ({note})")
if c in g:
    out["LOCK_ETA_C"], note = c, "registered"
elif c_fb in g:
    out["LOCK_ETA_C"], note = c_fb, f"the registered fallback for {c}"
elif near(c):
    out["LOCK_ETA_C"], note = near(c), f"SUBSTITUTED for {c}: P1 at it is computed after registration"
else:
    out["LOCK_ETA_C"], note = "none", f"{c}, {c_fb} and every clock within 15 MHz unsupported: dropped"
print(f"# LOCK_ETA_C {out['LOCK_ETA_C']} ({note})")
for f in probe:
    out[f"LOCK_PROBE_{f}"] = "supported" if f in g else "unsupported"
    print(f"# LOCK_PROBE_{f} {out[f'LOCK_PROBE_{f}']} (a plan's floor lock; probed below)")
for k, v in out.items():
    print(f"{k}={v}")
PY
  then
    ledger "prelude REFUSED: the supported-clock list could not be read (lock-plan.txt)"; return "$EXIT_REFUSED"
  fi
  grep '^#' "$S/lock-plan.txt" | sed 's/^# //' | tee "$S/supported-locks.txt"
  grep -v '^#' "$S/lock-plan.txt" > "$D/locks.env"
  local line; while IFS= read -r line; do ledger "lock plan: $line"; done < <(grep '^# LOCK_' "$S/lock-plan.txt" | sed 's/^# //')
  if ! grep -qx 'LOCK_TIMED_OK=1' "$D/locks.env"; then
    ledger "prelude REFUSED: $LOCK_TIMED MHz is not a supported graphics clock: every step locks it"; return "$EXIT_REFUSED"
  fi
  nvidia-smi -q -d POWER > "$S/power-at-start.txt"   # both scopes, before any change
  # A2 (tileA2-gates rule P-W): the card's max enforceable GPU limit must reach the plan's; no fallback
  local pmax; pmax="$(nvidia-smi --query-gpu=power.max_limit --format=csv,noheader,nounits | head -1 | tr -d ' ')"   # A2
  if ! awk -v p="$pmax" -v w="$POWER_LIMIT_W" 'BEGIN { exit !(p + 0 >= w - 0.5) }'; then   # A2
    ledger "prelude REFUSED: power.max_limit reads '$pmax' W, under the registered $POWER_LIMIT_W W (tileA2-gates P-W: the session stops, no fallback to 700 W)"; return "$EXIT_REFUSED"   # A2
  fi   # A2
  ledger "prelude: power.max_limit $pmax W, at or over the registered $POWER_LIMIT_W W"   # A2
  sudo -n nvidia-smi -pm 1 || { ledger "prelude REFUSED: persistence mode not set"; return "$EXIT_REFUSED"; }
  sudo -n nvidia-smi -pl "$POWER_LIMIT_W" -sc 0 \
    || { ledger "prelude REFUSED: the GPU power limit was not set to $POWER_LIMIT_W W"; return "$EXIT_REFUSED"; }
  if ! s1_power_settle; then   # C2: the plan's limit or nothing (no fallback)
    echo "C2_POWER_${POWER_LIMIT_W}=refused" >> "$D/c2-power.env"   # C2
    ledger "prelude REFUSED: the power limit never read back as $POWER_LIMIT_W W on both readings within ${PL_WAIT_S} s (C2: no fallback limit, the session stops; power-set-unsettled-${POWER_LIMIT_W}.txt)"   # C2
    return "$EXIT_REFUSED"   # C2
  fi   # C2
  echo "C2_POWER_W=$POWER_LIMIT_W" >> "$D/c2-power.env"   # C2: the limit every unit ran under
  nvidia-smi --query-gpu=persistence_mode,clocks.sm,clocks.max.sm,power.limit,power.default_limit,clocks_event_reasons.active \
    --format=csv | tee "$S/clocks-at-start.txt"
  local pm pl
  IFS=', ' read -r pm pl < <(nvidia-smi --query-gpu=persistence_mode,power.limit --format=csv,noheader,nounits | head -1)
  if [[ "$pm" != Enabled ]]; then
    ledger "prelude REFUSED: persistence_mode reads '$pm', not Enabled"; return "$EXIT_REFUSED"
  fi
  if ! awk -v p="$pl" -v w="$POWER_LIMIT_W" 'BEGIN { exit !(p + 0 > w - 1 && p + 0 < w + 1) }'; then
    ledger "prelude REFUSED: power.limit reads '$pl' W, not $POWER_LIMIT_W"; return "$EXIT_REFUSED"
  fi
  # A plan's floor locks besides 1710: lock each once, read it back, reset. A lock
  # that does not hold is recorded, and its floor units refuse in 0 minutes.
  local pf res
  for pf in ${PROBE_LOCKS//,/ }; do
    if grep -qx "LOCK_PROBE_$pf=supported" "$D/locks.env"; then
      if lock_and_check "$pf"; then res=held; else res=refused; fi
      sudo -n nvidia-smi -rgc >/dev/null || { ledger "prelude REFUSED: the reset after probing $pf MHz failed"; return "$EXIT_REFUSED"; }
    else
      res=unsupported
    fi
    echo "LOCK_PROBE_$pf=$res" >> "$D/locks.env"
    ledger "lock probe: $pf MHz $res"
  done
  # C2: the lock-hold probe in its cheapest form (S1's prelude probe, made a stop): the session lock itself
  if lock_and_check "$LOCK_TIMED"; then res=held; else res=refused; fi   # C2
  sudo -n nvidia-smi -rgc >/dev/null || { ledger "prelude REFUSED: the reset after probing the session lock $LOCK_TIMED MHz failed"; return "$EXIT_REFUSED"; }   # C2
  echo "LOCK_PROBE_$LOCK_TIMED=$res" >> "$D/locks.env"   # C2
  ledger "lock probe: the session lock $LOCK_TIMED MHz $res (C2)"   # C2
  if [[ "$res" != held ]]; then   # C2
    ledger "prelude REFUSED: the session lock $LOCK_TIMED MHz did not read back after -lgc (C2: a board that cannot hold it stops before any page)"; return "$EXIT_REFUSED"   # C2
  fi   # C2
  echo "A2_LOCK=$LOCK_TIMED" > "$D/a2-lock.env"   # A2: the session's lock until the lock-hold probe picks its rung
  ledger "prelude: persistence Enabled, power.limit $pl W, base tag $B, session lock $LOCK_TIMED MHz"   # A2
  return "$EXIT_DONE"
}

# ==========================================================================
# 3c.2 byte pages at the 1710 lock
# ==========================================================================
counters_dir() {
  local R f="$S/counters-dir.txt" tag=""
  # A plan unit's pages: a directory of the model and label's own, its name fixed at
  # first use as the session's is (a unit run past midnight keeps its date).
  [[ -n "$OUT_LABEL" ]] && { tag="$MODEL-$OUT_LABEL"; f="$S/counters-dir-$tag.txt"; }
  if [[ -s "$f" ]]; then R="$(cat "$f")"
  else R="$RESULTS_ROOT/$(date -u +%F)-$MOE_CARD${tag:+-$tag}-r3-counters"; echo "$R" > "$f"; fi
  mkdir -p "$R/base"
  printf '%s\n' "$R"
}

step_bytes() {
  local R F C L G tail_dropped=0
  if [[ -n "$INSTR_EVICT_A$INSTR_EVICT_B" ]]; then
    gate_ok bytes || return "$EXIT_REFUSED"
  fi
  R="$(counters_dir)"; F="$LOCK_TIMED"; L="$R/lock$F"
  C="$(census_for)" || return "$EXIT_REFUSED"
  mkdir -p "$L"
  [[ -s "$C" ]] || { ledger "bytes REFUSED: no census at $C (the preflight's PF6 writes it)"; return "$EXIT_REFUSED"; }
  local gs=() override=0 knobs=() x tk=()   # TILE
  # a read loop, not mapfile: a laptop's /bin/bash is 3.2 (the tests run there)
  while IFS= read -r x; do [[ -n "$x" ]] && knobs+=("$x"); done < <(knob_args)
  while IFS= read -r x; do [[ -n "$x" ]] && tk+=("$x"); done < <(tile_dcr_args)   # TILE
  while IFS= read -r x; do [[ -n "$x" ]] && tk+=("$x"); done < <(s1_dcr_args)   # S1
  (( ${#tk[@]} )) && ledger "bytes: tile ${tk[*]} through $(dcr_script) on every page of this unit"   # TILE
  (( ${#knobs[@]} )) && ledger "bytes: rental 2 knobs ${knobs[*]} on every page of this unit"
  if [[ -n "$BYTE_GROUPS" ]]; then
    override=1
    for G in ${BYTE_GROUPS//,/ }; do gs+=("$G"); done
    ledger "bytes: the registered list G = ${gs[*]} ($MODEL${OUT_LABEL:+, $OUT_LABEL}): no tail drop, no re-price, no base-clock control"
  else
    dropped bytes_tail && tail_dropped=1
    for G in "${BYTE_GS[@]}"; do
      [[ "$G" == 64 && "$tail_dropped" == 1 ]] || gs+=("$G")
    done
  fi
  (( tail_dropped )) && ledger "bytes: the G=64 page and the base-clock control are dropped (deadline)"
  ( set -o pipefail
    trap 'trap "" INT TERM HUP; moe_counter ncu --clock-control reset >/dev/null 2>&1; sudo -n nvidia-smi -rgc >/dev/null' EXIT
    trap 'exit 130' INT TERM HUP
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo "the GPU is in use: stop"; exit 2; }
    moe_counter ncu --clock-control reset
    MAX=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
    [ "$F" -le "$MAX" ] || { echo "F=$F MHz is above this card's maximum, $MAX MHz"; exit 2; }
    lock_and_check "$F" || exit 2
    # The re-price threshold is 8x7B's page time; another model's pages run
    # longer by its scale (8x22B's G=1 page took 583 s on 2026-09-28, 1.39x
    # 8x7B's 420 s threshold, and dropped a registered G=64 page until the
    # operator restored it).
    reprice_s=$(( BYTE_G1_REPRICE_S * $(model_scale_pct) / 100 ))
    for G in "${gs[@]}"; do
      if [ "$G" = 64 ] && [ -f "$D/bytes-tail-repriced" ]; then continue; fi
      LOG=$S/logs/r3c-$(uid)-g$G-lock$F.log
      t0=$(date +%s)
      # TILE: dcr_script is the counter route's wrapper when the unit names num-warps; tk its tile flags
      moe_counter "$PY_VLLM" "$(dcr_script)" ${tk[@]+"${tk[@]}"} $(: TILE) --run --family r3-arms --group-m "$G" \
        --model "$MODEL" --tiles "${BYTE_TREADS:-$TREADS}" $(: S1) --census "$C" --page-clock none --page-lock-mhz "$F" \
        ${knobs[@]+"${knobs[@]}"} --out "$L/r3c-g$G.json" 2>&1 | tee "$LOG"   # TILE
      rc=${PIPESTATUS[0]}
      secs=$(( $(date +%s) - t0 ))
      echo "G=$G exit $rc $(date -u +%T) ${secs} s" | tee -a "$S/logs/lock$F-pages-$(uid).status"
      failed=$(grep -E '^RESULT: VALIDITY [^ ]+ FAIL' "$LOG" | awk '{print $3}' | sort -u | tr '\n' ' ')
      case $rc in
        0|1) ;;
        # A WRITTEN INVALID PAGE GOES ON (2026-09-27). The draft stopped on any
        # INVALID but V7 alone at G >= 32; on the 2026-09-27 GH200 that stopped
        # the loop at G=1 on V10, a gate defect since fixed, and the G=2 page
        # still fails V7 on one cell at the noise edge. Every page keeps its
        # .ncu-rep and its gates re-score on the laptop (--analyse), so a page
        # written is data; only a page NOT written stops the loop.
        3) if [ -f "$L/r3c-g$G.json" ]; then
             echo "G=$G INVALID on ${failed:-no gate line}: the page is written, go on (re-scored later)"
           else
             echo "G=$G INVALID and no page written: the byte pages stop here; read $LOG"; exit 3
           fi ;;
        *) exit "$rc" ;;
      esac
      if [ "$override" = 0 ] && [ "$G" = 1 ] && [ "$secs" -gt "$reprice_s" ]; then
        touch "$D/bytes-tail-repriced"
        echo "G=1 took $secs s, over $reprice_s: the G=64 page and the base-clock control are dropped"
      fi
      # EVERY PAGE GOES TO GITHUB AS IT LANDS (2026-09-28): the Qwen2-57B GH200
      # vanished 55 min into this step and took every page with it, the step
      # pushing only at its end. Between pages nothing is measured (the page's
      # process has exited, the next has not started), so the rule holds.
      ledger "bytes: G=$G page exit $rc after ${secs} s"
      push_results "bytes: G=$G page exit $rc"
    done )
  local rc=$?
  ledger "bytes: lock block exit $rc ($(tr '\n' ';' < "$S/logs/lock$F-pages-$(uid).status" 2>/dev/null))"
  [[ -f "$D/bytes-tail-repriced" ]] && tail_dropped=1 \
    && ledger "bytes: G=1 took over its re-price threshold: the G=64 page and the base-clock control dropped"
  sleep 5
  nvidia-smi --query-gpu=clocks.sm,clocks.mem,clocks.max.sm,clocks_event_reasons.active,persistence_mode \
    --format=csv | tee "$S/clocks-after-lock-pages.txt"
  if (( override )) || (( ${#knobs[@]} )) || (( ${#tk[@]} )); then   # TILE
    ledger "bytes: no base-clock control (a registered --byte-groups list, or a knob unit)"
  elif (( ! tail_dropped )) && [[ "$MODEL" != "$DEFAULT_MODEL" ]]; then
    ledger "bytes: no base-clock control (it compares clock regimes on $DEFAULT_MODEL; no $MODEL prediction reads it)"
  elif (( ! tail_dropped )); then
    # the base-clock control: same board, commit and treads, at ncu's base clock
    moe_counter "$PY_VLLM" scripts/dram_counter_route.py --run --family r3-arms --group-m 2 \
      --model "$MODEL" --tiles "$TREADS" \
      --census "$C" --out "$R/base/r3c-g2.json" 2>&1 | tee "$S/logs/r3c-$(uid)-g2-base.log"
    ledger "bytes: base-clock control G=2 exit ${PIPESTATUS[0]}"
    moe_counter ncu --clock-control reset
  fi
  local timed=() id p
  [[ "$MODEL" == "$DEFAULT_MODEL" ]] || ledger "bytes: no C5 references (the 2026-09-25 timed pages are $DEFAULT_MODEL's)"
  for id in "${TIMED_REF_IDS[@]}"; do
    [[ "$MODEL" == "$DEFAULT_MODEL" && "$override" == 0 ]] || break
    for p in "$REPO/$TIMED_REF_DIR"/*"$id"/report.json; do
      if [[ -f "$p" ]]; then timed+=("$p"); else ledger "bytes: timed reference *$id is missing from the checkout"; fi
    done
  done
  local pages=()
  for p in "$L"/r3c-g*.json; do [[ -f "$p" ]] && pages+=("$p"); done
  if (( ${#pages[@]} )); then
    python3 scripts/dram_counter_route.py --analyse "${pages[@]}" \
      ${timed[@]+--timed-reference "${timed[@]}"} --out "$L/summary.json" 2>&1 | tee "$S/logs/analyse-$(uid)-lock$F.log"
    ledger "bytes: analyse of ${#pages[@]} pages exit ${PIPESTATUS[0]} ($L/summary.json)"
  else
    ledger "bytes: no page written, nothing to analyse"
  fi
  return "$rc"
}

# ==========================================================================
# 3c.3 calibrate, with no lock in force
# ==========================================================================
step_calibrate() {
  export MOE_RESULTS_DIR="$(results_dir)"
  local busy; busy="$(gpu_busy)"
  [[ -z "$busy" ]] || { ledger "calibrate REFUSED: the GPU is in use: $busy"; return "$EXIT_REFUSED"; }
  moe_counter ncu --clock-control reset; sudo -n nvidia-smi -rgc
  nvidia-smi -q -d POWER > "$S/power-before-calibrate.txt"
  "$PY_BASE" scripts/calibrate_hardware.py --publish --results-root "$RESULTS_ROOT" 2>&1 | tee "$S/logs/calibrate-$(uid).log"
  local rc=${PIPESTATUS[0]} Y="moe/bench/hardware/measured_$MOE_CARD.yaml" kind why
  grep '^RESULT:' "$S/logs/calibrate-$(uid).log" | cut -c1-160 | while IFS= read -r l; do ledger "calibrate: $l"; done
  if (( rc == EXIT_DONE )) && [[ -f "$Y" ]]; then
    kind=measured; why="calibrate exit 0"
  elif (( rc == EXIT_CLAIM_FAIL )) && grep -qE '^RESULT: CLAIM not_throttled FAIL' "$S/logs/calibrate-$(uid).log"; then
    kind=fallback; why="calibrate exit 1 with not_throttled FAILED"
  elif (( rc == EXIT_CLAIM_FAIL )) && [[ -f "$Y" ]]; then
    kind=measured
    why="calibrate exit 1, not_throttled did not fail (failed: $(grep -E '^RESULT: [A-Z]+ [^ ]+ FAIL' "$S/logs/calibrate-$(uid).log" | awk '{print $3}' | tr '\n' ' '))"
  else
    kind=fallback; why="calibrate exit $rc$([[ -f "$Y" ]] || echo ', no ruler written')"
  fi
  if [[ "$kind" == measured ]]; then
    cp "$Y" "$S/"
    printf 'RULER_KIND=measured\nRULER_ARGS=()\n' > "$D/ruler.env"
    ledger "calibrate: the ruler stands ($why): $Y"
  else
    rm -f "$Y"
    { printf 'RULER_KIND=fallback\nRULER_ARGS=('; printf '%q ' "${RULER_FALLBACK[@]}"; printf ')\n'; } > "$D/ruler.env"
    ledger "calibrate: FALLBACK ($why): the ruler is removed and R3 takes ${RULER_FALLBACK[*]}, the 2026-09-25 ruler's"
  fi
  git status --porcelain | tee "$S/git-after-calibrate.txt"   # the ruler, untracked, when it stands
  return "$rc"
}

# ==========================================================================
# 3c.3 timed R3 at 1710 (P2, P5)
# ==========================================================================
#: R3's dry run of one design, logged, then checked: prints the reason when the
#: design may not run.
r3_dry() {   # LOG PLANNED TAG G ARGS...
  local log="$1" planned="$2" tag="$3" g="$4"; shift 4
  # TILE: r3_script is scripts/r3_tile.py when the unit names num-warps
  "$PY_VLLM" "$(r3_script)" $(: TILE) "$@" --group-m "$g" --session-tag "$tag" --dry-run \
    > "$log" 2>&1   # TILE
  grep -E '^REFUSED:|^ *ridge |n_decl = [0-9]+ against|retracted +tread|predicted peak' "$log" | cut -c1-160
  local why
  if why="$(check_dry "$log" "$planned")"; then return 0; fi
  ledger "dry run G=$g treads $planned may not lock: $why ($log)"
  return 1
}

step_timed() {
  export MOE_RESULTS_DIR="$(results_dir)"
  if [[ -n "$DECLARED_COPIES" ]]; then
    mkdir -p "$MOE_RESULTS_DIR"
    { echo "declared-copies $DECLARED_COPIES (plan key, rental 4): R3 --declared-copies $DECLARED_COPIES"
      [[ "$DECLARED_COPIES" == "$COUNTER_COPIES" ]] \
        || echo "NOT JOINABLE TO COUNTER BYTES: every counter page declares R3's $COUNTER_COPIES copies (validate_counter_plan), so no byte page shares this page's grid"
    } > "$MOE_RESULTS_DIR/DECLARED_COPIES.txt"
    ledger "timed: declared-copies $DECLARED_COPIES on every page of this unit$([[ "$DECLARED_COPIES" == "$COUNTER_COPIES" ]] || echo ', labelled NOT JOINABLE TO COUNTER BYTES')"
  fi
  load_ruler
  # rental 3: a unit's timed-groups / timed-treads replace G=8, 32 at treads 6 and drop G=3
  local TT; TT="$(timed_treads)"
  local own=0; [[ -n "$TIMED_GROUPS$TIMED_TREADS" ]] && own=1
  local tk=() x
  while IFS= read -r x; do [[ -n "$x" ]] && tk+=("$x"); done < <(timed_knob_args)
  while IFS= read -r x; do [[ -n "$x" ]] && tk+=("$x"); done < <(tile_r3_args)   # TILE
  local R6=("${R3_BASE[@]}" --treads "$TT" ${RULER_ARGS[@]+"${RULER_ARGS[@]}"} ${tk[@]+"${tk[@]}"})
  local DRY_SLIP=()
  if [[ -n "$HISTOGRAM" ]]; then
    # rental 5: a histogram page (its cells, not a ladder); the dry run's tread is its deepest n
    TT="$(hist_deepest "$HISTOGRAM")"
    [[ -n "$TT" ]] || { ledger "timed REFUSED: the histogram page $HISTOGRAM is unreadable"; return "$EXIT_REFUSED"; }
    own=1
    local ds=()
    while IFS= read -r x; do [[ -n "$x" ]] && ds+=("$x"); done < <(dry_slip)
    R6=("${R3_BASE[@]}" --histogram "$HISTOGRAM" --arms "${R3_ARMS:-native,shared}"
        ${RULER_ARGS[@]+"${RULER_ARGS[@]}"} ${tk[@]+"${tk[@]}"})
    DRY_SLIP=(${ds[@]+"${ds[@]}"})
    ledger "timed: histogram page $HISTOGRAM, arms ${R3_ARMS:-native,shared}, deepest tread $TT$( (( ${#tk[@]} )) && echo ", knobs ${tk[*]}")$( (( ${#ds[@]} )) && echo ", scorer G1 rule $G1_RULE (locked_r3 --min-clean-repeats 0)")"
  fi
  local R8=("${R3_BASE[@]}" --treads 8 ${RULER_ARGS[@]+"${RULER_ARGS[@]}"})
  local T2 T5=""
  T2="$(fresh_tag "$B-p2")" || { ledger "timed REFUSED: no fresh session tag for $B-p2"; return "$EXIT_REFUSED"; }
  if (( ! own )); then
    T5="$(fresh_tag "$B-p5")" || { ledger "timed REFUSED: no fresh session tag for $B-p5"; return "$EXIT_REFUSED"; }
  fi
  local busy; busy="$(gpu_busy)"
  [[ -z "$busy" ]] || { ledger "timed REFUSED: the GPU is in use: $busy"; return "$EXIT_REFUSED"; }
  moe_counter ncu --clock-control reset; sudo -n nvidia-smi -rgc
  (( own )) && ledger "timed: G = $(timed_groups) at treads $TT, no G=3 page (timed-groups / timed-treads)"
  local p2=() G
  for G in $(timed_groups); do
    r3_dry "$S/logs/r3-$(uid)-g$G-t$TT-dry.log" "$TT" "$T2-lock$LOCK_TIMED" "$G" "${R6[@]}" ${DRY_SLIP[@]+"${DRY_SLIP[@]}"} && p2+=("$G")
  done
  local p5=0
  (( own )) || { r3_dry "$S/logs/r3-$(uid)-g3-t8-dry.log" 8 "$T5-lock$LOCK_TIMED" 3 "${R8[@]}" && p5=1; }
  ( set -o pipefail
    trap 'trap "" INT TERM HUP; sudo -n nvidia-smi -rgc >/dev/null' EXIT
    trap 'exit 130' INT TERM HUP
    rc2=0; rc5=0
    if (( ${#p2[@]} )); then
      # TILE: lr3_script hands locked_r3 --r3-script scripts/r3_tile.py when the unit names num-warps
      python3 scripts/locked_r3.py --session-tag "$T2" --groups "${p2[@]}" --locks "$LOCK_TIMED" $(lr3_cap) $(lr3_slip) $(lr3_minclean) $(lr3_script) $(: TILE) -- "${R6[@]}" 2>&1 \
        | tee -i "$S/logs/locked_r3-$(uid)-$T2.log"
      rc2=${PIPESTATUS[0]}
      echo "P2 locked_r3 exit $rc2" >> "$D/timed-results"
      (( rc2 >= 4 )) && exit "$rc2"   # a crash, or the card left locked
    else
      echo "P2 skipped: no design passed its dry run" >> "$D/timed-results"; rc2=3
    fi
    if (( own )); then
      echo "P5 not run: this unit names its own groups (timed-groups / timed-treads)" >> "$D/timed-results"
    elif (( p5 )); then
      python3 scripts/locked_r3.py --session-tag "$T5" --groups 3 --locks "$LOCK_TIMED" $(lr3_cap) -- "${R8[@]}" 2>&1 \
        | tee -i "$S/logs/locked_r3-$(uid)-$T5.log"
      rc5=${PIPESTATUS[0]}
      echo "P5 locked_r3 exit $rc5" >> "$D/timed-results"
    else
      echo "P5 skipped: its design did not pass its dry run" >> "$D/timed-results"; rc5=3
    fi
    exit $(( rc2 > rc5 ? rc2 : rc5 )) )
  local rc=$?
  sleep 5
  nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks_event_reasons.active,power.limit --format=csv \
    | tee "$S/clocks-after-timed.txt"
  local l; while IFS= read -r l; do ledger "timed: $l"; done < <(
    cat "$D/timed-results" 2>/dev/null; ledger_lines "$S/locked-r3/$T2/status" ${T5:+"$S/locked-r3/$T5/status"})
  : > "$D/timed-results"
  return "$rc"
}

# ==========================================================================
# 3c.4 held locks below 1710 (P1)
# ==========================================================================
step_eta() {
  export MOE_RESULTS_DIR="$RESULTS_ROOT/gaps-$MOE_CARD"
  load_ruler; load_locks
  local R6=("${R3_BASE[@]}" --treads 6 ${RULER_ARGS[@]+"${RULER_ARGS[@]}"})
  local ok4=0 ok2=0
  r3_dry "$S/logs/r3-eta-$(uid)-g4-dry.log" 6 "$B-eta-check" 4 "${R6[@]}" && ok4=1
  r3_dry "$S/logs/r3-eta-$(uid)-g2-dry.log" 6 "$B-eta-check" 2 "${R6[@]}" && ok2=1
  # 1410 first: it alone answers the falsifier. Each lock is its own run, since
  # locked_r3.py stops at the first lock that holds.
  local specs=() gs
  if [[ "$LOCK_ETA_A" != none ]]; then
    gs=""; (( ok4 )) && gs="4"; (( ok2 )) && gs="$gs 2"
    [[ -n "${gs# }" ]] && specs+=("$LOCK_ETA_A:${gs# }")
  fi
  (( ok4 )) && [[ "$LOCK_ETA_B" != none ]] && specs+=("$LOCK_ETA_B:4")
  if (( ok4 )) && [[ "$LOCK_ETA_C" != none ]]; then
    if dropped eta_1605; then ledger "eta: the $LOCK_ETA_C lock is dropped (deadline)"
    else specs+=("$LOCK_ETA_C:4"); fi
  fi
  (( ${#specs[@]} )) || { ledger "eta: no lock to run (dry runs or supported clocks)"; return "$EXIT_INVALID"; }
  local busy; busy="$(gpu_busy)"
  [[ -z "$busy" ]] || { ledger "eta REFUSED: the GPU is in use: $busy"; return "$EXIT_REFUSED"; }
  : > "$D/eta-results.new"
  ( set -o pipefail
    trap 'trap "" INT TERM HUP; sudo -n nvidia-smi -rgc >/dev/null' EXIT
    trap 'exit 130' INT TERM HUP
    moe_counter ncu --clock-control reset; sudo -n nvidia-smi -rgc
    worst=0
    for spec in "${specs[@]}"; do
      F=${spec%%:*}; GS=${spec#*:}
      for try in 1 2; do
        tag=$(fresh_tag "$B-eta$F") || { echo "no fresh tag for $B-eta$F"; exit 4; }
        # shellcheck disable=SC2086
        python3 scripts/locked_r3.py --session-tag "$tag" --locks "$F" --groups $GS $(lr3_cap) -- "${R6[@]}" 2>&1 \
          | tee -i "$S/logs/eta-$(uid)-lock$F-$tag.log"
        rc=${PIPESTATUS[0]}
        echo "$F $rc $tag" >> "$D/eta-results.new"
        # Exit 1 this far under 1710 more likely means a lock left over than the
        # power cap: reset, record the clocks, and try once more under a new tag.
        if (( rc == 1 && try == 1 )); then
          sudo -n nvidia-smi -rgc
          nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks_event_reasons.active --format=csv \
            > "$S/clocks-after-eta-$tag.txt"
          continue
        fi
        break
      done
      (( rc > worst )) && worst=$rc
      (( rc >= 4 )) && exit "$rc"
    done
    exit "$worst" )
  local rc=$?
  cat "$D/eta-results.new" >> "$D/eta-results"
  sleep 5
  nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks_event_reasons.active --format=csv | tee "$S/clocks-after-eta.txt"
  local l tag; while IFS= read -r l; do
    tag="${l##* }"; ledger "eta: lock ${l%% *} locked_r3 exit $(echo "$l" | awk '{print $2}') ($tag)"
    ledger_lines "$S/locked-r3/$tag/status" | while IFS= read -r x; do ledger "eta: $x"; done
  done < "$D/eta-results.new"
  rm -f "$D/eta-results.new"
  return "$rc"
}

# ==========================================================================
# 3c.5 floor counters (P6)
# ==========================================================================
step_floor() {
  local R C F=$LOCK_TIMED fk=() x; R="$(counters_dir)"
  if [[ -n "$FLOOR_GROUPS" && -n "$FLOOR_LOCK_MHZ" ]]; then
    F="$FLOOR_LOCK_MHZ"
    if [[ "$F" != "$LOCK_TIMED" ]]; then
      local probe=""; probe="$(sed -n "s/^LOCK_PROBE_$F=//p" "$D/locks.env" 2>/dev/null | tail -1)"
      if [[ "$probe" != held ]]; then
        ledger "floor REFUSED: the prelude's probe of the $F MHz lock reads ${probe:-nothing (never probed)} (locks.env LOCK_PROBE_$F), not held"
        return "$EXIT_REFUSED"
      fi
    fi
  fi
  if [[ -n "$FLOOR_GROUPS" ]]; then
    while IFS= read -r x; do [[ -n "$x" ]] && fk+=("$x"); done < <(floor_knob_args)
  fi
  C="$(census_for)" || return "$EXIT_REFUSED"
  [[ -s "$C" ]] || { ledger "floor REFUSED: no census at $C"; return "$EXIT_REFUSED"; }
  : > "$D/floor-results"
  ( set -o pipefail
    SMI=
    trap 'trap "" INT TERM HUP; [ -n "${SMI:-}" ] && kill "$SMI" 2>/dev/null; moe_counter ncu --clock-control reset >/dev/null 2>&1; sudo -n nvidia-smi -rgc >/dev/null' EXIT
    trap 'exit 130' INT TERM HUP
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo "the GPU is in use: stop"; exit 2; }
    moe_counter ncu --clock-control reset
    sudo -n nvidia-smi -rgc >/dev/null || exit 2
    worst=0
    # 2 is refused (census, commit or door) and would refuse every capture;
    # 3 is INVALID with the file written, and the next capture still answers.
    capture() {   # NAME, then dram_counter_route.py's arguments
      local name=$1; shift
      moe_counter "$PY_VLLM" scripts/dram_counter_route.py --run --family r3-arms --model "$MODEL" "$@" 2>&1 \
        | tee "$S/logs/$name-$(uid).log"
      local rc=${PIPESTATUS[0]}
      echo "$name exit $rc" >> "$D/floor-results"
      (( rc > worst )) && worst=$rc
      case $rc in 0|3) return 0 ;; *) exit "$rc" ;; esac
    }
    if [ -n "$FLOOR_GROUPS" ]; then
      ft=(); [ -n "$FLOOR_TREADS" ] && ft=(--floor-treads ${FLOOR_TREADS//,/ })
      if [ "$FLOOR_BASE" != 0 ]; then
        for g in ${FLOOR_GROUPS//,/ }; do
          capture "r3f-g$g" --group-m "$g" --census "$C" --floor "${ft[@]}" ${fk[@]+"${fk[@]}"} \
            --out "$R/r3f-g$g.json"
        done
      else
        echo "no base-clock capture: floor-base=0 (the registered design takes the $F MHz lock only)" >> "$D/floor-results"
      fi
      moe_counter ncu --clock-control reset
      MAX=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
      [ "$F" -le "$MAX" ] || { echo "F=$F MHz is above this card's maximum, $MAX MHz"; exit 2; }
      lock_and_check "$F" || exit 2
      for g in ${FLOOR_GROUPS//,/ }; do
        capture "r3f-g$g-lock$F" --group-m "$g" --census "$C" --floor "${ft[@]}" --floor-clock none \
          --floor-lock-mhz "$F" ${fk[@]+"${fk[@]}"} --out "$R/r3f-g$g-lock$F.json"
      done
      echo "registered floor design: groups $FLOOR_GROUPS, treads ${FLOOR_TREADS:-default}" >> "$D/floor-results"
      exit "$worst"
    fi
    capture r3f-g64 --group-m 64 --census "$C" --floor --out "$R/r3f-g64.json"
    passes=$(grep -oE -- '- [0-9]+ pass(es)?' "$S/logs/r3f-g64-$(uid).log" | awk '{print $2}' | sort -n | tail -1)
    if [ "$MODEL" != "$DEFAULT_MODEL" ]; then
      # another model's registered floor test reads G=64 at base and at the lock
      echo "r3f-g2 and r3f-g64-unlocked left out: no $MODEL prediction reads them" >> "$D/floor-results"
    elif [ "${passes:-0}" -gt "$FLOOR_MAX_PASSES" ]; then
      echo "r3f-g2 dropped: the G=64 capture took $passes replay passes a launch, over $FLOOR_MAX_PASSES" >> "$D/floor-results"
    else
      capture r3f-g2 --group-m 2 --census "$C" --floor --out "$R/r3f-g2.json"
    fi
    moe_counter ncu --clock-control reset
    # a RECORD, scoring nothing: P7 is registered for the duty-0.25 timed regime
    if [ "$MODEL" = "$DEFAULT_MODEL" ]; then
      nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.max.sm,power.draw,power.limit,clocks_event_reasons.active \
        --format=csv,noheader,nounits -lms 100 > "$S/logs/r3f-g64-unlocked-$(uid).smi.csv" 2>&1 & SMI=$!
      capture r3f-g64-unlocked --group-m 64 --census "$C" --floor --floor-clock none \
        --out "$R/r3f-g64-unlocked.json"
      kill "$SMI" 2>/dev/null; wait "$SMI" 2>/dev/null; SMI=
    fi
    moe_counter ncu --clock-control reset
    MAX=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
    [ "$F" -le "$MAX" ] || { echo "F=$F MHz is above this card's maximum, $MAX MHz"; exit 2; }
    lock_and_check "$F" || exit 2
    capture "r3f-g64-lock$F" --group-m 64 --census "$C" --floor --floor-clock none --floor-lock-mhz "$F" \
      --out "$R/r3f-g64-lock$F.json"
    exit "$worst" )
  local rc=$?
  sleep 5
  nvidia-smi --query-gpu=clocks.sm,clocks.mem,clocks.max.sm,clocks_event_reasons.active --format=csv \
    | tee "$S/clocks-after-floor.txt"
  local l; while IFS= read -r l; do ledger "floor: $l"; done < "$D/floor-results"
  return "$rc"
}

# ==========================================================================
# 3c.6 timed R3 to tread 9 (P3, P4)
# ==========================================================================
step_deep() {
  export MOE_RESULTS_DIR="$(results_dir)"
  load_ruler
  local R9=("${R3_BASE[@]}" --treads 9 ${RULER_ARGS[@]+"${RULER_ARGS[@]}"})
  local T; T="$(fresh_tag "$B-deep")" || { ledger "deep REFUSED: no fresh session tag for $B-deep"; return "$EXIT_REFUSED"; }
  moe_counter ncu --clock-control reset; sudo -n nvidia-smi -rgc
  local busy; busy="$(gpu_busy)"
  [[ -z "$busy" ]] || { ledger "deep REFUSED: the GPU is in use: $busy"; return "$EXIT_REFUSED"; }
  local gs=() G
  for G in 4 2; do   # G=4 first: a run cut short still answers P4
    r3_dry "$S/logs/r3-deep-$(uid)-g$G-dry.log" 9 "$T-lock$LOCK_TIMED" "$G" "${R9[@]}" && gs+=("$G")
  done
  (( ${#gs[@]} )) || { ledger "deep: no design passed its dry run"; return "$EXIT_INVALID"; }
  ( set -o pipefail
    trap 'trap "" INT TERM HUP; sudo -n nvidia-smi -rgc >/dev/null' EXIT
    trap 'exit 130' INT TERM HUP
    python3 scripts/locked_r3.py --session-tag "$T" --locks "$LOCK_TIMED" --groups "${gs[@]}" $(lr3_cap) -- "${R9[@]}" 2>&1 \
      | tee -i "$S/logs/locked_r3-$(uid)-$T.log" )
  local rc=$?
  ledger "deep: locked_r3 exit $rc ($T, G = ${gs[*]})"
  sleep 5
  nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks_event_reasons.active --format=csv | tee "$S/clocks-after-deep.txt"
  local l; while IFS= read -r l; do ledger "deep: $l"; done < <(ledger_lines "$S/locked-r3/$T/status")
  # P3 and P4 off the two pages: read only when both show every VALIDITY gate
  # passing and the 9-copy declaration.
  "$PY_VLLM" - "$MOE_RESULTS_DIR/private_weight_reference" "$T-lock$LOCK_TIMED" > "$S/logs/deep-check-$(uid)-$T.txt" 2>&1 <<'PY'
import glob, json, statistics, sys
root, tag = sys.argv[1:3]; ms = {}
for p in sorted(glob.glob(f"{root}/*/report.json")):
    j = json.load(open(p))
    if j.get("session_tag") != tag:
        continue
    G = j["pinned"]["GROUP_SIZE_M"]
    bad = [f"{g.get('tag')} {g.get('verdict')}" for g in j["gates"]
           if g.get("kind") == "VALIDITY" and g.get("verdict") != "PASS"]
    print(f"G={G} treads 1..{j['treads'][-1]} declared {j['copies_declared']} "
          f"VALIDITY not PASS: {bad or 'none'}")
    ms.update({(G, r["tiles"]): r["ms_p50"] for r in j["treads_table"] if r["arm"] == "shared"})
for G in (2, 4):
    t = [ms[(G, n)] for n in range(1, 10) if (G, n) in ms]
    print(f"G={G} SHARED ms", " ".join(f"{v:.4f}" for v in t),
          "| increments", " ".join(f"{b - a:.3f}" for a, b in zip(t, t[1:])))
if all((g, n) in ms for g in (2, 4) for n in (3, 5, 7, 9)):
    print("P3 G=2 minus G=4 at n = 3 5 7 9:",
          " ".join(f"{ms[(2, n)] - ms[(4, n)]:.3f}" for n in (3, 5, 7, 9)), "ms")
if all((4, n) in ms for n in range(2, 10)):
    inc = [ms[(4, n + 1)] - ms[(4, n)] for n in range(2, 9)]
    print(f"P4 G=4 n=8 to 9: {inc[-1]:.3f} ms against the median of the others, "
          f"{statistics.median(inc[:-1]):.3f}")
PY
  while IFS= read -r l; do ledger "deep: $l"; done < "$S/logs/deep-check-$(uid)-$T.txt"
  return "$rc"
}

# ==========================================================================
# 3c.7 R1 in lock mode, last
# ==========================================================================
step_r1lock() {
  export MOE_RESULTS_DIR="$RESULTS_ROOT/gaps-$MOE_CARD"
  load_locks
  # Every lock R1 takes held in 3c.4; a lock that did not is replaced by the
  # ones that did. Before 3c.4 has run, its planned locks.
  local locks=("$LOCK_TIMED") f
  if [[ -s "$D/eta-results" ]]; then
    for f in "$LOCK_ETA_B" "$LOCK_ETA_A"; do
      [[ "$f" != none ]] && awk -v f="$f" '$1 == f && $2 == 0 { found = 1 } END { exit !found }' "$D/eta-results" \
        && locks+=("$f")
    done
  else
    for f in "$LOCK_ETA_B" "$LOCK_ETA_A"; do [[ "$f" != none ]] && locks+=("$f"); done
  fi
  if (( ${#locks[@]} < 2 )) || (( locks[${#locks[@]} - 1] > R1_SPAN_MAX_LOW )); then
    ledger "r1lock: no held lock at or under $R1_SPAN_MAX_LOW MHz (locks ${locks[*]}): R1 cannot span its claim; not run"
    return "$EXIT_INVALID"
  fi
  local gs=() G
  dropped r1_g4 && ledger "r1lock: R1 at G=4 is dropped (deadline)" || gs+=(4)
  dropped r1_g1 && ledger "r1lock: R1 at G=1 is dropped (deadline)" || gs+=(1)
  (( ${#gs[@]} )) || return "$EXIT_INVALID"
  local tag; tag="$(fresh_tag "$B-r1lock")" || { ledger "r1lock REFUSED: no fresh session tag for $B-r1lock"; return "$EXIT_REFUSED"; }
  local R1=("${R1_BASE[@]}" --session-tag "$tag")
  for G in "${gs[@]}"; do
    "$PY_VLLM" scripts/clock_elasticity.py --dry-run --group-m "$G" --lock-clocks "${locks[@]}" "${R1[@]}" \
      > "$S/logs/r1lock-$(uid)-g$G.plan.log" 2>&1
    echo "plan G=$G exit $? (2 expected)"
  done
  nvidia-smi -q -d POWER > "$S/power-before-r1lock.txt"
  : > "$D/r1-results"
  ( set -o pipefail
    trap 'trap "" INT TERM HUP; sudo -n nvidia-smi -rgc >/dev/null' EXIT
    trap 'exit 130' INT TERM HUP
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo "the GPU is in use: stop"; exit 2; }
    moe_counter ncu --clock-control reset
    sudo -n nvidia-smi -rgc
    for G in "${gs[@]}"; do
      "$PY_VLLM" scripts/clock_elasticity.py --group-m "$G" --lock-clocks "${locks[@]}" "${R1[@]}" \
        2>&1 | tee "$S/logs/r1lock-$(uid)-g$G.log"
      rc=${PIPESTATUS[0]}; echo "r1lock G=$G exit $rc (locks ${locks[*]})" >> "$D/r1-results"
      [ "$rc" -le 1 ] || exit "$rc"   # 0 DONE, 1 CLAIM_FAIL: go on; 2, 3, 4: stop
    done )
  local rc=$?
  sleep 5
  nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks_event_reasons.active --format=csv | tee "$S/clocks-after-r1lock.txt"
  local l; while IFS= read -r l; do ledger "r1lock: $l"; done < "$D/r1-results"
  return "$rc"
}

# ==========================================================================
# launchfloor: R3's eager timer against its host (2026-10-01, docs/registered)
# ==========================================================================
#: Where one model's launch-floor files go: its own directory, under the unit's label.
launch_dir() {
  local f="$S/launch-dir${OUT_LABEL:+-$OUT_LABEL}.txt" R
  if [[ -s "$f" ]]; then R="$(cat "$f")"
  else R="$RESULTS_ROOT/$(date -u +%F)-$MOE_CARD-launch-floor${OUT_LABEL:+-$OUT_LABEL}"; echo "$R" > "$f"; fi
  printf '%s\n' "$R/$MODEL"
}

step_launchfloor() {
  local R; R="$(launch_dir)"
  # TWO PROCESSES (rental 2, 2026-10-01): the timed phase, whose process has the
  # profiler guarded off and measures the host first, then the trace phase. Rental 1
  # ran both in one process and its traces' profiler state confounded the timing.
  local LF=(--model "$MODEL" --treads "$LF_TREADS" --modes "$LF_MODES"
            --arms "$LF_ARMS" --group-m "$LF_GROUP_M" --duty 0.25 --repeats 3 --seed 0 --out "$R")
  # rental 3 (2026-10-05): lf-trace=0 runs the timed phase alone (the wall-time check)
  local phases=(timed); [[ "$LF_TRACE" == 0 ]] || phases+=(trace)
  local TIMED=("${LF[@]}" --phase timed) TRACE=("${LF[@]}" --phase trace --trace-treads "$LF_TRACE")
  local busy; busy="$(gpu_busy)"
  [[ -z "$busy" ]] || { ledger "launchfloor REFUSED: the GPU is in use: $busy"; return "$EXIT_REFUSED"; }
  # each phase's own dry run first: 2 is a plan printed, anything else is a refusal
  local ph drc
  for ph in "${phases[@]}"; do
    if [[ "$ph" == timed ]]; then "$PY_VLLM" scripts/launch_floor.py "${TIMED[@]}" --dry-run > "$S/logs/launchfloor-$(uid)-$ph-dry.log" 2>&1
    else "$PY_VLLM" scripts/launch_floor.py "${TRACE[@]}" --dry-run > "$S/logs/launchfloor-$(uid)-$ph-dry.log" 2>&1; fi
    drc=$?
    if (( drc != EXIT_REFUSED )) || grep -q '^REFUSED:' "$S/logs/launchfloor-$(uid)-$ph-dry.log"; then
      ledger "launchfloor REFUSED for $MODEL: its $ph dry run exited $drc ($(grep -m1 '^REFUSED:' "$S/logs/launchfloor-$(uid)-$ph-dry.log" | cut -c1-160))"
      return "$EXIT_REFUSED"
    fi
  done
  mkdir -p "$R"
  ( set -o pipefail
    trap 'trap "" INT TERM HUP; sudo -n nvidia-smi -rgc >/dev/null' EXIT
    trap 'exit 130' INT TERM HUP
    moe_counter ncu --clock-control reset >/dev/null 2>&1 || true
    MAX=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
    [ "$LOCK_TIMED" -le "$MAX" ] || { echo "F=$LOCK_TIMED MHz is above this card's maximum, $MAX MHz"; exit 2; }
    lock_and_check "$LOCK_TIMED" || exit 2
    "$PY_VLLM" scripts/launch_floor.py "${TIMED[@]}" 2>&1 | tee "$S/logs/launchfloor-$(uid)-timed.log"
    rt=${PIPESTATUS[0]}
    echo "timed phase exit $rt" >> "$S/logs/launchfloor-$(uid).phases"
    (( rt >= 4 )) && exit "$rt"
    rr=0
    if [[ "$LF_TRACE" == 0 ]]; then
      echo "trace phase not run (lf-trace=0)" >> "$S/logs/launchfloor-$(uid).phases"
    else
      "$PY_VLLM" scripts/launch_floor.py "${TRACE[@]}" 2>&1 | tee "$S/logs/launchfloor-$(uid)-trace.log"
      rr=${PIPESTATUS[0]}
      echo "trace phase exit $rr" >> "$S/logs/launchfloor-$(uid).phases"
    fi
    exit $(( rt > rr ? rt : rr )) )
  local rc=$?
  sleep 5
  nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks_event_reasons.active --format=csv \
    | tee "$S/clocks-after-launchfloor-$MODEL.txt"
  ledger "launchfloor: $MODEL exit $rc ($R)"
  return "$rc"
}

# ==========================================================================
# rental 4 (2026-10-06): rulers, gpubench, perturb, stamps
# ==========================================================================
#: A plan-only unit's own directory, <date>-<card>-<what>-<label> under $RESULTS_ROOT.
unit_out() {   # WHAT
  local f="$S/unit-dir-$1${OUT_LABEL:+-$OUT_LABEL}.txt" R
  if [[ -s "$f" ]]; then R="$(cat "$f")"
  else R="$RESULTS_ROOT/$(date -u +%F)-$MOE_CARD-$1${OUT_LABEL:+-$OUT_LABEL}"; echo "$R" > "$f"; fi
  mkdir -p "$R"
  printf '%s\n' "$R"
}
#: An instrumented unit runs only when the perturb unit wrote PASS for its variant.
gate_ok() {   # STEP
  local k line
  k="GATE_$(printf '%s' "$VARIANT_ID" | tr -c 'A-Za-z0-9' '_')"
  line="$(grep -E "^$k=" "$D/instr-gate.env" 2>/dev/null | tail -1)"
  if [[ "${line#*=}" != PASS* ]]; then
    ledger "$1 REFUSED: the perturbation gate for ${VARIANT_ID:-(no variant)} reads '${line#*=}' (instr-gate.env): an instrumented unit runs only on a PASS"
    return 1
  fi
  return 0
}
#: Lock 1710, run one tool under it, reset (the subshell's trap); the tool's exit.
locked_run() {   # LOG CMD...
  local log="$1"; shift
  ( set -o pipefail
    trap 'trap "" INT TERM HUP; sudo -n nvidia-smi -rgc >/dev/null' EXIT
    trap 'exit 130' INT TERM HUP
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo "the GPU is in use: stop"; exit 2; }
    moe_counter ncu --clock-control reset >/dev/null 2>&1 || true
    MAX=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
    [ "$LOCK_TIMED" -le "$MAX" ] || { echo "F=$LOCK_TIMED MHz is above this card's maximum, $MAX MHz"; exit 2; }
    lock_and_check "$LOCK_TIMED" || exit 2
    "$@" 2>&1 | tee "$log"
    exit "${PIPESTATUS[0]}" )
}
#: A tool's own dry run first: 2 with no REFUSED line may go on, anything else refuses.
dry_ok() {   # NAME LOG CMD...
  local name="$1" log="$2"; shift 2
  "$@" --dry-run > "$log" 2>&1
  local drc=$?
  if (( drc != EXIT_REFUSED )) || grep -q '^REFUSED:' "$log"; then
    ledger "$name REFUSED: its dry run exited $drc ($(grep -m1 '^REFUSED:' "$log" | cut -c1-160))"
    return 1
  fi
  return 0
}

# ==========================================================================   # TILE
# the tile stage: the compile-only tile census (scripts/tile_census.py)   # TILE
# ==========================================================================   # TILE
step_tilecensus() {   # TILE
  local out="$RESULTS_ROOT/$(date -u +%F)-$MOE_CARD-tile-census-$(uid)" log="$S/logs/tilecensus-$(uid).log"   # TILE
  local busy; busy="$(gpu_busy)"   # TILE
  [[ -z "$busy" ]] || { ledger "tilecensus REFUSED: the GPU is in use: $busy"; return "$EXIT_REFUSED"; }   # TILE
  mkdir -p "$out"   # TILE
  "$PY_VLLM" scripts/tile_census_a2.py --tiles "$CENSUS_TILES" --models "$CENSUS_MODELS" --dry-run $(: A2) \
    > "$S/logs/tilecensus-$(uid)-dry.log" 2>&1   # TILE A2
  "$PY_VLLM" scripts/tile_census_a2.py --tiles "$CENSUS_TILES" --models "$CENSUS_MODELS" --out "$out/tile-census.json" 2>&1 | tee "$log"   # TILE A2
  local rc=${PIPESTATUS[0]} l   # TILE
  while IFS= read -r l; do ledger "tilecensus: $l"; done < <(grep '^RESULT:' "$log" | cut -c1-160)   # TILE
  ledger "tilecensus: exit $rc ($out/tile-census.json)"   # TILE
  return "$rc"   # TILE
}   # TILE
# TILE: end of the tile census step
# ==========================================================================   # A2
# tile rental A2: the gates and the lock-hold probe (docs/registered/2026-10-20-tileA2-gates-gh200)   # A2
# ==========================================================================   # A2
step_gate() {   # A2
  local rec="$D/gate-$GATE_NAME-$LOCK_TIMED.json" log="$S/logs/gate-$GATE_NAME-$(uid).log" rc   # A2
  "$PY_BASE" scripts/tileA2_gate.py "$GATE_NAME" --session "$S" --results "$RESULTS_ROOT" --repo "$REPO" \
    --lock "$LOCK_TIMED" --record "$rec" > "$log" 2>&1   # A2
  rc=$?   # A2
  ledger "gate $GATE_NAME at $LOCK_TIMED MHz: $(grep -m1 '^RESULT: GATE' "$log" | cut -c1-220) (exit $rc)"   # A2
  return "$rc"   # A2
}   # A2
step_lockprobe() {   # A2
  local L entry model hist G short rc pw log failed   # A2
  for L in ${LADDER//,/ }; do   # A2
    if ! grep -qx "LOCK_PROBE_$L=held" "$D/locks.env"; then   # A2
      ledger "lockprobe: rung $L MHz FAILS: the prelude's probe reads '$(grep "^LOCK_PROBE_$L=" "$D/locks.env" | cut -d= -f2)'"; continue   # A2
    fi   # A2
    LOCK_TIMED="$L"; failed=0   # A2
    ledger "lockprobe: rung $L MHz: hw_rulers, then ${PROBES//,/ }, then the OLMoE f128 counter page (P5)"   # A2
    nvidia-smi -q -d POWER > "$S/power-lockprobe-$L-before.txt" 2>&1   # A2: both scopes
    nvidia-smi -q -d POWER -lms 1000 > "$S/logs/lockprobe-$L-power.log" 2>&1 &   # A2: the module scope too, each second
    pw=$!   # A2
    OUT_LABEL="lp$L"; step_rulers || ledger "lockprobe: hw_rulers at $L exited non-zero (DESCRIPTIVE only)"   # A2
    for entry in ${PROBES//,/ } "olmoe-1b-7b:-:64:f"; do   # A2: the last is P5's counter page
      IFS=: read -r model hist G short <<< "$entry"   # A2
      MODEL="$model"; OUT_LABEL="lp$L$short"   # A2
      if [[ "$short" == f ]]; then   # A2: tileA2-gates P5, the plan's f128 unit at the rung's lock
        HISTOGRAM=""; R3_ARMS=""; BYTE_GROUPS="$G"; SLIP_POLICY=""; G1_RULE=""; DECLARED_COPIES=""   # A2
        step_bytes; ledger "lockprobe: $model $OUT_LABEL counter page exit $?"   # A2
      else   # A2
        HISTOGRAM="$hist"; TIMED_GROUPS="$G"; TIMED_TREADS=""   # A2
        R3_ARMS=native,shared; SLIP_POLICY=cell; G1_RULE=r7-median-run3; DECLARED_COPIES=9   # A2
        R3_BASE=(--model "$MODEL" --block-m "${TILE_BLOCK_M:-32}" --repeats 9 --duty 0.25 --seed 0 --declared-copies 9)   # A2
        step_timed; ledger "lockprobe: $model $OUT_LABEL timed exit $?"   # A2
      fi   # A2
      if ! "$PY_BASE" scripts/tileA2_gate.py probe --partial --results "$RESULTS_ROOT" --session "$S" --lock "$L"           > "$S/logs/gate-probe-$L-partial.log" 2>&1; then   # A2
        ledger "lockprobe: rung $L MHz: $OUT_LABEL fails the rule, the rung's remaining pages are not run (early stop)"   # A2
        failed=1; break   # A2
      fi   # A2
    done   # A2
    kill "$pw" 2>/dev/null; wait "$pw" 2>/dev/null   # A2
    nvidia-smi -q -d POWER > "$S/power-lockprobe-$L-after.txt" 2>&1   # A2
    log="$S/logs/gate-probe-$L.log"   # A2
    "$PY_BASE" scripts/tileA2_gate.py probe --results "$RESULTS_ROOT" --session "$S" --lock "$L"       --record "$D/gate-probe-$L.json" > "$log" 2>&1   # A2
    rc=$?   # A2
    (( failed && rc == 0 )) && rc=$EXIT_INVALID   # A2: an early stop never passes
    ledger "lockprobe: rung $L MHz: $(grep -m1 '^RESULT: GATE' "$log" | cut -c1-220) (exit $rc)"   # A2
    if (( rc == 0 )); then   # A2
      echo "A2_LOCK=$L" > "$D/a2-lock.env"   # A2
      ledger "lockprobe: rung $L MHz HOLDS: the session's lock is $L MHz (a2-lock.env)"; return 0   # A2
    fi   # A2
  done   # A2
  ledger "lockprobe: no rung of ${LADDER//,/ } MHz held (tileA2-gates G-PROBE)"   # A2
  return "$EXIT_INVALID"   # A2
}   # A2
step_powerproof() {   # S1
  local R; R="$(unit_out powerproof)"   # S1
  local busy; busy="$(gpu_busy)"   # S1
  [[ -z "$busy" ]] || { ledger "powerproof REFUSED: the GPU is in use: $busy"; return "$EXIT_REFUSED"; }   # S1
  local P=("$PY_VLLM" scripts/power_proof_s1.py --seconds "$POWERPROOF_S" --out "$R/gemm.json")   # S1
  dry_ok powerproof "$S/logs/powerproof-$(uid)-dry.log" "${P[@]}" || return "$EXIT_REFUSED"   # S1
  moe_counter ncu --clock-control reset >/dev/null 2>&1 || true   # S1
  sudo -n nvidia-smi -rgc >/dev/null || { ledger "powerproof REFUSED: the clock reset failed"; return "$EXIT_REFUSED"; }   # S1
  nvidia-smi -q -d POWER > "$R/power-before.txt" 2>&1   # S1
  nvidia-smi --query-gpu=timestamp,power.draw,enforced.power.limit,power.limit,clocks.sm,clocks.max.sm,clocks_event_reasons.active,temperature.gpu $(: S1) \
    --format=csv,noheader,nounits -lms 100 > "$R/smi.csv" 2>&1 &   # S1
  local smi=$! rc   # S1
  "${P[@]}" 2>&1 | tee "$S/logs/powerproof-$(uid).log"   # S1
  rc=${PIPESTATUS[0]}   # S1
  kill "$smi" 2>/dev/null; wait "$smi" 2>/dev/null   # S1
  nvidia-smi -q -d POWER > "$R/power-after.txt" 2>&1   # S1
  ledger "powerproof: GEMM exit $rc, $(wc -l < "$R/smi.csv" | tr -d ' ') nvidia-smi samples ($R)"   # S1
  return "$rc"   # S1
}   # S1
step_rulers() {
  local R; R="$(unit_out rulers)"
  local T=("$PY_VLLM" scripts/hw_rulers.py --out "$R" --lock-mhz "$LOCK_TIMED")
  dry_ok rulers "$S/logs/rulers-$(uid)-dry.log" "${T[@]}" || return "$EXIT_REFUSED"
  locked_run "$S/logs/rulers-$(uid).log" "${T[@]}"
  local rc=$?
  ledger "rulers: exit $rc ($R/rulers.json)"
  return "$rc"
}

step_gpubench() {
  local R; R="$(unit_out gpubench)"
  local G=("$PY_BASE" scripts/gpubench.py --dest "$GPUBENCH_DEST" --out "$R" --sm 90
           --lock-mhz "$LOCK_TIMED" --l2-mib "$GPUBENCH_L2_MIB")
  dry_ok gpubench "$S/logs/gpubench-$(uid)-dry.log" "${G[@]}" || return "$EXIT_REFUSED"
  locked_run "$S/logs/gpubench-$(uid).log" "${G[@]}"
  local rc=$?
  ledger "gpubench: exit $rc (third-party GPL-3.0 code at its pin, outside the checkout; $R/constants.json)"
  return "$rc"
}

step_perturb() {
  local R V="$D/instr-variants.json"; R="$(unit_out perturb)"
  [[ -s "$V" ]] || { ledger "perturb REFUSED: no $V (no instrumented unit after this one)"; return "$EXIT_REFUSED"; }
  # its own Triton cache (build-r4-review G1): env.sh's shared one would hide this unit's compiles
  local P=(env "TRITON_CACHE_DIR=$R/triton-cache" "$PY_VLLM" scripts/instr_probe_c3.py --mode perturb --variants "$V" --out "$R")   # C3
  dry_ok perturb "$S/logs/perturb-$(uid)-dry.log" "${P[@]}" || return "$EXIT_REFUSED"
  locked_run "$S/logs/perturb-$(uid).log" "${P[@]}"
  local rc=$?
  if [[ -s "$R/gate.env" ]]; then
    cat "$R/gate.env" >> "$D/instr-gate.env"
    local l; while IFS= read -r l; do ledger "perturb: $l"; done < "$R/gate.env"
  else
    ledger "perturb: no gate.env written: every instrumented unit after it refuses"
  fi
  return "$rc"
}

step_regcheck() {
  local R V="$D/instr-variants.json"; R="$(unit_out regcheck)"
  [[ -s "$V" ]] || { ledger "regcheck REFUSED: no $V (no instrumented unit in the plan)"; return "$EXIT_REFUSED"; }
  local P=(env "TRITON_CACHE_DIR=$R/triton-cache" "$PY_VLLM" scripts/instr_probe_c3.py --mode regcheck --variants "$V" --out "$R")   # C3
  dry_ok regcheck "$S/logs/regcheck-$(uid)-dry.log" "${P[@]}" || return "$EXIT_REFUSED"
  # compile only: no lock, nothing timed (each variant's kernels are built, never measured)
  "${P[@]}" 2>&1 | tee "$S/logs/regcheck-$(uid).log"
  local rc=${PIPESTATUS[0]}
  if [[ -s "$R/regcheck.env" ]]; then
    cat "$R/regcheck.env" >> "$D/instr-regcheck.env"
    local l; while IFS= read -r l; do ledger "regcheck: $l"; done < "$R/regcheck.env"
  else
    ledger "regcheck: no regcheck.env written: every variant-listing stamps unit refuses"
  fi
  return "$rc"
}

#: C3 (owner OD-1, 2026-10-26): a histogram stamps unit runs on its HARD line (every gate leg but timing); a FAIL   # C3
#: on the timing legs alone keeps the unit, its pages labelled PERTURBED by the probe   # C3
c3_hard_ok() {   # C3
  local k g h   # C3
  k="$(printf '%s' "$VARIANT_ID" | tr -c 'A-Za-z0-9' '_')"   # C3
  h="$(grep -E "^HARD_$k=" "$D/instr-gate.env" 2>/dev/null | tail -1)"; g="$(grep -E "^GATE_$k=" "$D/instr-gate.env" 2>/dev/null | tail -1)"   # C3
  if [[ "${h#*=}" != PASS* ]]; then   # C3
    ledger "stamps REFUSED: the gate's hard legs for ${VARIANT_ID:-(no variant)} read '${h#*=}' (instr-gate.env): occupancy, SASS and the upstream file must pass"   # C3
    return 1   # C3
  fi   # C3
  [[ "${g#*=}" == PASS* ]] || ledger "stamps: the overhead gate for $VARIANT_ID reads '${g#*=}': the pages run PERTURBED (owner OD-1, C3)"   # C3
  return 0   # C3
}   # C3
#: rental 5: the first of a stamps unit's variants whose regcheck and perturb lines read PASS
choose_variant() {
  local id k g r
  for id in $VARIANT_IDS; do
    k="$(printf '%s' "$id" | tr -c 'A-Za-z0-9' '_')"
    r="$(grep -E "^REGCHECK_$k=" "$D/instr-regcheck.env" 2>/dev/null | tail -1)"
    g="$(grep -E "^GATE_$k=" "$D/instr-gate.env" 2>/dev/null | tail -1)"
    if [[ "${r#*=}" == PASS* && "${g#*=}" == PASS* ]]; then printf '%s\n' "$id"; return 0; fi
  done
  return 1
}

step_stamps() {
  if [[ -n "$INSTR_VARIANTS" ]]; then
    local chosen; chosen="$(choose_variant)" || {
      ledger "stamps REFUSED: none of $VARIANT_IDS reads PASS on both its regcheck and perturb lines"; return "$EXIT_REFUSED"; }
    VARIANT_ID="$chosen"
    local R; R="$(unit_out "instr-$MODEL")"
    printf 'id=%s\nspec=%s\n' "$chosen" "$(r5_variant_spec "${chosen##*-}")" > "$R/CHOSEN_VARIANT.txt"
    ledger "stamps: $chosen chosen (the first of $VARIANT_IDS passing regcheck and perturb)"
    local P=(env "TRITON_CACHE_DIR=$R/triton-cache" "$PY_VLLM" scripts/instr_probe_c3.py --mode stamps --variants "$D/instr-variants.json"   # C3
             --variant "$chosen" --gate "$D/instr-gate.env" --regcheck "$D/instr-regcheck.env" --out "$R")
    dry_ok stamps "$S/logs/stamps-$(uid)-$chosen-dry.log" "${P[@]}" || return "$EXIT_REFUSED"
    locked_run "$S/logs/stamps-$(uid)-$chosen.log" "${P[@]}"
    local rc=$?
    ledger "stamps: $chosen exit $rc ($R)"
    return "$rc"
  fi
  if [[ -n "$HISTOGRAM" ]]; then   # C3
    c3_hard_ok || return "$EXIT_REFUSED"   # C3
  else   # C3
  gate_ok stamps || return "$EXIT_REFUSED"
  fi   # C3
  local R; R="$(unit_out "instr-$MODEL")"
  local P=(env "TRITON_CACHE_DIR=$R/triton-cache" "$PY_VLLM" scripts/instr_probe_c3.py --mode stamps --variants "$D/instr-variants.json"   # C3
           --variant "$VARIANT_ID" --gate "$D/instr-gate.env" --regcheck "$D/instr-regcheck.env" --out "$R")   # C3
  dry_ok stamps "$S/logs/stamps-$(uid)-$VARIANT_ID-dry.log" "${P[@]}" || return "$EXIT_REFUSED"
  locked_run "$S/logs/stamps-$(uid)-$VARIANT_ID.log" "${P[@]}"
  local rc=$?
  ledger "stamps: $VARIANT_ID exit $rc ($R)"
  return "$rc"
}

# ==========================================================================
# the orchestrator
# ==========================================================================
push_results() {   # MESSAGE [FINAL_EXIT]
  (( NO_PUSH )) && return 0
  local p="$MOE_HOME/repo/scripts/vm_results_push.sh"
  [[ -f "$p" ]] || p="$(dirname "$SELF")/vm_results_push.sh"
  local rc=0
  timeout -k 30 900 bash "$p" push --message "$1" ${2:+--final "$2"} || rc=$?
  (( rc == 0 )) || ledger "push of '$1' exited $rc (results stay on the VM; the next push carries them)"
  return 0
}

#: Between steps: clear both kinds of lock, wait for an idle card, record.
hygiene() {
  moe_counter ncu --clock-control reset >/dev/null 2>&1 || true
  if ! sudo -n nvidia-smi -rgc >/dev/null 2>&1; then
    ledger "HYGIENE: nvidia-smi -rgc failed: the card may still be locked"; return 1
  fi
  local waited=0 busy
  while busy="$(gpu_busy)"; [[ -n "$busy" ]]; do
    if (( waited >= BUSY_WAIT_S )); then
      ledger "HYGIENE: the GPU is still in use ${BUSY_WAIT_S} s after the step: $busy"; return 2
    fi
    sleep 10; waited=$(( waited + 10 ))
  done
  ledger "clocks: $(nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.limit,clocks_event_reasons.active \
    --format=csv,noheader 2>&1 | head -1)"
  return 0
}

left_min() { echo $(( (DEADLINE - $(now_s)) / 60 - RESERVE_MIN )); }

est_now() {   # the step's estimate after its drops
  local s="$1" m d; m="$(step_est "$s")"
  for d in "${DROPS[@]}"; do
    [[ "$(drop_step "$d")" == "$s" ]] && dropped "$d" && m=$(( m - $(drop_min "$d") ))
  done
  echo "$m"
}

#: Before step i: drop in the draft's order while the rest overruns the time left.
budget() {   # INDEX
  local i="$1" j need left d
  while :; do
    need=0
    for (( j = i; j < ${#PLAN[@]}; j++ )); do need=$(( need + $(est_now "${PLAN[$j]}") )); done
    left="$(left_min)"
    (( need <= left )) && return 0
    local applied=0
    for d in "${DROPS[@]}"; do
      dropped "$d" && continue
      local ds; ds="$(drop_step "$d")"
      for (( j = i; j < ${#PLAN[@]}; j++ )); do
        if [[ "${PLAN[$j]}" == "$ds" ]]; then
          echo "$d" >> "$D/drops"
          ledger "DROPPED $(drop_what "$d"): ${need} min of steps left, ${left} min to the deadline"
          applied=1; break
        fi
      done
      (( applied )) && break
    done
    (( applied )) || return 0
  done
}

STEP_PID=""
on_signal() {
  local sig="$1"
  trap '' INT TERM HUP
  ledger "STOPPED by SIG$sig during ${CURRENT:-no step}: stopping the step, then the reset"
  [[ -n "$STEP_PID" ]] && { kill -TERM "$STEP_PID" 2>/dev/null; wait "$STEP_PID" 2>/dev/null; }
  hygiene || true
  push_results "stopped by SIG$sig during ${CURRENT:-no step}" 143
  exit 143
}

run_step() {   # NAME CAP_S
  local name="$1" cap="$2" rc
  local log="$S/logs/driver-$name-$(uid).log"
  timeout -k 240 "$cap" bash "$SELF" --step "$name" >> "$log" 2>&1 < /dev/null &
  STEP_PID=$!
  wait "$STEP_PID"; rc=$?
  STEP_PID=""
  return "$rc"
}

print_plan() {
  local s total=0
  echo "THE GH200 MODEL-TEST SESSION (docs/LAMBDA.md section 3c), unattended"
  [[ "$MODEL" == "$DEFAULT_MODEL" ]] || echo "  MODEL $MODEL: a cross-model test (docs/registered/README.md); its own census, estimates x$(model_scale_pct)%, no eta or R1 unless --steps asks"
  echo "  owner, 2026-09-26: byte treads $TREADS; calibrate after the byte pages; GPU power limit"
  echo "  ${POWER_LIMIT_W} W (-sc 0); r570 upgraded to 580.105.08 before setup (scripts/vm_run.sh)"
  printf '  %-10s %4s %4s  %s\n' step est cap what
  for s in "${PLAN[@]}"; do
    printf '  %-10s %4s %4s  %s\n' "$s" "$(step_est "$s")" "$(step_cap "$s")" "$(step_what "$s")"
    total=$(( total + $(step_est "$s") ))
  done
  echo "  $total min of steps, estimated, plus setup_vm.sh (~30 min) when --setup; about $(( (total + 30) / 60 )) h"
  echo "  deadline: $([[ -n "$DEADLINE" ]] && utc_of "$DEADLINE" || echo 'none given (the run itself needs --deadline)')"
  echo "  drop order when time runs short: R1 at G=1, R1 at G=4, the 1605 lock, the G=64 byte page and"
  echo "  the base-clock control; a step whose own estimate no longer fits is skipped"
  echo "  pushes: after every step, to the run's branch (scripts/vm_results_push.sh)$( (( NO_PUSH )) && echo ': OFF (--no-push)')"
  echo "COMMANDS (\$S, \$R and \$B as on the VM; R3 takes ${RULER_FALLBACK[*]} only if calibrate fails):"
  echo "  bytes   moe_counter \$PY_VLLM scripts/dram_counter_route.py --run --family r3-arms --group-m G --tiles $TREADS"
  echo "            --census \$S/census.json --page-clock none --page-lock-mhz $LOCK_TIMED --out \$R/lock$LOCK_TIMED/r3c-gG.json"
  echo "            for G in ${BYTE_GS[*]}, under nvidia-smi -lgc $LOCK_TIMED,$LOCK_TIMED; then G=2 at ncu's base clock"
  echo "  cal     \$PY_BASE scripts/calibrate_hardware.py --publish --results-root \$RESULTS_ROOT"
  echo "  timed   python3 scripts/locked_r3.py --session-tag \$B-p2 --groups $(timed_groups) --locks $LOCK_TIMED -- ${R3_BASE[*]} --treads $(timed_treads)"
  [[ -n "$TIMED_GROUPS$TIMED_TREADS" ]] && echo "          no G=3 page (timed-groups / timed-treads)" || \
  echo "          python3 scripts/locked_r3.py --session-tag \$B-p5 --groups 3 --locks $LOCK_TIMED -- ${R3_BASE[*]} --treads 8"
  echo "  eta     python3 scripts/locked_r3.py --session-tag \$B-eta<F> --locks <F> --groups <G> -- ${R3_BASE[*]} --treads 6"
  echo "            for 1410:4 2, 1500:4, 1605:4 (1590 if 1605 is not supported)"
  echo "  floor   moe_counter \$PY_VLLM scripts/dram_counter_route.py --run --family r3-arms --floor --census \$S/census.json"
  [[ -z "$FLOOR_GROUPS" ]] || echo "          registered floor design: G = ${FLOOR_GROUPS//,/ }, treads ${FLOOR_TREADS:-default} (${FLOOR_TREADS:+--floor-treads ${FLOOR_TREADS//,/ }}), base and lock $LOCK_TIMED; no r3f-g2 or unlocked record"
  echo "            G=64 and G=2 at base; G=64 --floor-clock none (unlocked record); G=64 --floor-clock none --floor-lock-mhz $LOCK_TIMED"
  echo "            NATIVE at treads $(floor_treads) for $MODEL (dram_counter_route.r3_floor_treads, from the model)"
  echo "  deep    python3 scripts/locked_r3.py --session-tag \$B-deep --locks $LOCK_TIMED --groups 4 2 -- ${R3_BASE[*]} --treads 9"
  echo "  r1lock  \$PY_VLLM scripts/clock_elasticity.py --group-m G --lock-clocks 1710 1500 1410 ${R1_BASE[*]} --session-tag \$B-r1lock"
  echo "            for G in 4 1"
  echo "Every timed design runs R3's own --dry-run first and locks only when it prints no REFUSED line,"
  echo "'n_decl = 9 against n_max', a retracted tread of at least the treads planned, and a ruler."
}

# ---- a plan of several models (--plan FILE) ------------------------------
U_MODEL=(); U_STEP=(); U_OPTS=(); U_EST=(); U_CAP=(); U_LABEL=(); U_GROUP=(); U_DEPENDS=(); U_NODROP=()
#: The plan file: the checkout's copy once setup has run (the commit measured is
#: the authority), else the path as given, else the copy vm_run.sh put beside
#: this driver for the dry run before setup.
resolve_plan() {
  local f="$1" c
  for c in "$MOE_HOME/repo/$f" "$f" "$(dirname "$SELF")/$(basename "$f")"; do
    [[ -f "$c" ]] && { printf '%s\n' "$c"; return 0; }
  done
  return 1
}
plan_refuse() { echo "--plan: line $1: $2" >&2; return 1; }
load_plan() {   # FILE
  local f="$1" line ln=0 kv k v model step est cap label opts seen="" key group seen_perturb=0 instr_after=0
  local depends seen_regcheck=0 seen_timed=0 groups_seen=" " nodrop
  local -a w
  while IFS= read -r line || [[ -n "$line" ]]; do
    ln=$(( ln + 1 )); line="${line%%#*}"
    read -r -a w <<< "$line"
    (( ${#w[@]} )) || continue
    model="${w[0]}"; step="${w[1]:-}"; est=""; cap=""; label=""; opts=""; group=""; depends=""; nodrop=""
    is_step "$step" || { plan_refuse "$ln" "no step '$step' (${STEPS[*]})"; return 1; }
    if [[ "$step" == prelude ]]; then
      [[ "$model" == - && ${#U_STEP[@]} -eq 0 ]] \
        || { plan_refuse "$ln" "the prelude is the first unit, '- prelude', and only once"; return 1; }
    elif [[ "$model" == - && ( "$step" == rulers || "$step" == gpubench || "$step" == perturb || "$step" == regcheck || "$step" == tilecensus || "$step" == lockprobe || "$step" == gate || "$step" == powerproof ) ]]; then   # TILE A2 S1
      (( ${#U_STEP[@]} )) || { plan_refuse "$ln" "the first unit is '- prelude'"; return 1; }
    else
      (( ${#U_STEP[@]} )) || { plan_refuse "$ln" "the first unit is '- prelude'"; return 1; }
      [[ "$model" =~ ^[a-z0-9][a-z0-9.-]*$ ]] || { plan_refuse "$ln" "'$model' is not a model name"; return 1; }
    fi
    for kv in "${w[@]:2}"; do
      k="${kv%%=*}"; v="${kv#*=}"
      [[ "$kv" == *=* && -n "$v" ]] || { plan_refuse "$ln" "'$kv' is not key=value"; return 1; }
      case "$k" in
        label) [[ "$v" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { plan_refuse "$ln" "label $v"; return 1; }; label="$v" ;;
        drop-group) [[ "$v" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { plan_refuse "$ln" "drop-group $v"; return 1; }; group="$v" ;;
        depends) [[ "$v" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { plan_refuse "$ln" "depends $v"; return 1; }
                 [[ "$groups_seen" == *" $v "* ]] || { plan_refuse "$ln" "depends=$v names no drop-group of an earlier unit"; return 1; }
                 depends="$v" ;;
        nodrop) [[ "$v" == 1 ]] || { plan_refuse "$ln" "nodrop $v: 1 or leave it out"; return 1; }; nodrop=1 ;;
        slip-policy) [[ "$v" =~ ^(page|cell)$ ]] || { plan_refuse "$ln" "slip-policy $v: page or cell"; return 1; }
          opts+=" $k=$v" ;;
        g1-rule) [[ "$v" =~ ^(r5-any|r7-median-run3)$ ]] || { plan_refuse "$ln" "g1-rule $v: r5-any or r7-median-run3"; return 1; }
          opts+=" $k=$v" ;;
        histogram)
          [[ "$v" =~ ^[A-Za-z0-9_./-]+[.]json$ && "$v" != /* && "$v" != *..* ]] || { plan_refuse "$ln" "histogram $v: a repo-relative .json path"; return 1; }
          [[ -f "$(dirname "$SELF")/../$v" || -f "$v" || -f "$MOE_HOME/repo/$v" ]] || { plan_refuse "$ln" "histogram $v: no such file"; return 1; }
          opts+=" $k=$v" ;;
        arms) [[ "$v" =~ ^(native|shared)(,(native|shared))?$ ]] || { plan_refuse "$ln" "arms $v: native and/or shared (PRIVATE cannot run a histogram page)"; return 1; }
          opts+=" $k=$v" ;;
        instr-variants) [[ "$v" =~ ^v[123](,v[123])*$ ]] || { plan_refuse "$ln" "instr-variants $v: from v1,v2,v3 (moe.instrumented.R5_VARIANTS)"; return 1; }
          opts+=" $k=$v" ;;
        est|cap) [[ "$v" =~ ^[0-9]+$ ]] || { plan_refuse "$ln" "$k $v: minutes"; return 1; }
                 [[ "$k" == est ]] && est="$v" || cap="$v" ;;
        byte-groups|floor-groups|floor-treads|lf-treads|lf-trace|timed-groups|stamp-groups|stamp-treads)
          [[ "$v" =~ ^[0-9]+(,[0-9]+)*$ ]] || { plan_refuse "$ln" "$k $v: a comma-separated list of integers"; return 1; }
          opts+=" $k=$v" ;;
        lf-modes) [[ "$v" =~ ^[A-Z0-9]+(,[A-Z0-9]+)*$ ]] || { plan_refuse "$ln" "lf-modes $v"; return 1; }; opts+=" $k=$v" ;;
        num-stages|block-k|slot-pad-rows|floor-lock-mhz|lf-group-m)
          [[ "$v" =~ ^[0-9]+$ ]] || { plan_refuse "$ln" "$k $v: a whole number"; return 1; }
          opts+=" $k=$v" ;;
        declared-copies|instr-every|instr-marks)
          [[ "$v" =~ ^[1-9][0-9]*$ ]] || { plan_refuse "$ln" "$k $v: a whole number of at least 1"; return 1; }
          opts+=" $k=$v" ;;
        instr-evict-a|instr-evict-b)
          [[ "$v" =~ ^(first|last|none)$ ]] || { plan_refuse "$ln" "$k $v: first, last or none"; return 1; }
          opts+=" $k=$v" ;;
        instr-stamps)
          [[ "$v" =~ ^(cta|iter|ends)$ ]] || { plan_refuse "$ln" "instr-stamps $v: cta, iter or ends"; return 1; }   # C3
          opts+=" $k=$v" ;;
        stamp-arms) [[ "$v" =~ ^(native|shared|private)(,(native|shared|private))*$ ]] || { plan_refuse "$ln" "stamp-arms $v: arms from native,shared,private"; return 1; }
          opts+=" $k=$v" ;;
        timed-treads) [[ "$v" =~ ^[1-9]$ ]] || { plan_refuse "$ln" "timed-treads $v: a tread 1..9 (R3's counter ladder)"; return 1; }
          opts+=" $k=$v" ;;
        lf-arms) [[ "$v" =~ ^(native|shared|private)(,(native|shared|private))*$ ]] || { plan_refuse "$ln" "lf-arms $v: arms from native,shared,private"; return 1; }
          opts+=" $k=$v" ;;
        partition-metrics|floor-base|floor-shape-metrics|floor-null-kernel)
          [[ "$v" =~ ^[01]$ ]] || { plan_refuse "$ln" "$k $v: 0 or 1"; return 1; }
          opts+=" $k=$v" ;;
        block-m) [[ "$v" =~ ^(16|32|64|128)$ ]] || { plan_refuse "$ln" "block-m $v: 16, 32, 64 or 128 (vLLM's default configs)"; return 1; }   # TILE
          opts+=" $k=$v" ;;   # TILE
        block-n) [[ "$v" =~ ^(64|128)$ ]] || { plan_refuse "$ln" "block-n $v: 64 or 128 (vLLM's default configs)"; return 1; }   # TILE
          opts+=" $k=$v" ;;   # TILE
        num-warps) [[ "$v" =~ ^(4|8)$ ]] || { plan_refuse "$ln" "num-warps $v: 4 or 8 (vLLM's default configs)"; return 1; }   # TILE
          opts+=" $k=$v" ;;   # TILE
        tiles) [[ "$v" =~ ^[0-9]+x[0-9]+x[0-9]+w[48]s[0-9]+(,[0-9]+x[0-9]+x[0-9]+w[48]s[0-9]+)*$ ]] || { plan_refuse "$ln" "tiles $v: BMxBNxBKw<warps>s<stages>[,...]"; return 1; }   # TILE
          opts+=" $k=$v" ;;   # TILE
        models) [[ "$v" =~ ^[a-z0-9][a-z0-9.-]*(,[a-z0-9][a-z0-9.-]*)*$ ]] || { plan_refuse "$ln" "models $v: model names, comma-separated"; return 1; }   # TILE
          opts+=" $k=$v" ;;   # TILE
        lock-mhz) [[ "$v" =~ ^[0-9]{3,4}$ ]] || { plan_refuse "$ln" "lock-mhz $v: a clock in MHz"; return 1; }; opts+=" $k=$v" ;;   # A2
        power-limit-w) [[ "$v" =~ ^(700|900)$ ]] || { plan_refuse "$ln" "power-limit-w $v: 700 or 900"; return 1; }; opts+=" $k=$v" ;;   # A2
        ladder) [[ "$v" =~ ^[0-9]{3,4}(,[0-9]{3,4})*$ ]] || { plan_refuse "$ln" "ladder $v: locks in MHz, comma-separated"; return 1; }; opts+=" $k=$v" ;;   # A2
        probes) local pre='[a-z0-9][a-z0-9.-]*:[A-Za-z0-9_./-]+[.]json:[0-9]+:[a-z]'   # A2
          [[ "$v" =~ ^$pre(,$pre)*$ && "$v" != *..* ]] || { plan_refuse "$ln" "probes $v: model:repo-relative.json:G:letter[,...]"; return 1; }; opts+=" $k=$v" ;;   # A2
        gate) [[ "$v" =~ ^(calibrate|census|band|f64w8)$ ]] || { plan_refuse "$ln" "gate $v: calibrate, census, band or f64w8"; return 1; }; opts+=" $k=$v" ;;   # A2
        unless-lock) [[ "$v" =~ ^[0-9]{3,4}$ ]] || { plan_refuse "$ln" "unless-lock $v: a clock in MHz"; return 1; }; opts+=" $k=$v" ;;   # A2
        byte-treads) [[ "$v" =~ ^[1-9](,[1-9])*$ ]] || { plan_refuse "$ln" "byte-treads $v: treads 1..9, comma-separated"; return 1; }; opts+=" $k=$v" ;;   # S1
        replay) [[ "$v" == application ]] || { plan_refuse "$ln" "replay $v: application (kernel replay is the route's own: leave the key out)"; return 1; }; opts+=" $k=$v" ;;   # S1
        stall-metrics) [[ "$v" == 1 ]] || { plan_refuse "$ln" "stall-metrics $v: 1 or leave it out"; return 1; }; opts+=" $k=$v" ;;   # S1
        *) plan_refuse "$ln" "no key '$k' (label est cap drop-group depends nodrop slip-policy g1-rule histogram arms instr-variants byte-groups floor-groups floor-treads lf-treads lf-modes lf-trace num-stages block-k slot-pad-rows partition-metrics floor-lock-mhz floor-base floor-shape-metrics floor-null-kernel timed-groups timed-treads lf-arms lf-group-m declared-copies instr-evict-a instr-evict-b instr-stamps instr-every instr-marks stamp-groups stamp-treads stamp-arms block-m block-n num-warps tiles models lock-mhz power-limit-w ladder probes gate unless-lock byte-treads replay stall-metrics)"; return 1 ;;   # TILE A2 S1
      esac
    done
    # rental 2's keys belong to one step each; a floor key needs a registered design
    for kv in ${opts}; do
      k="${kv%%=*}"
      case "$k" in
        slot-pad-rows|partition-metrics)
          [[ "$step" == bytes ]] || { plan_refuse "$ln" "$k belongs to a bytes unit, not $step"; return 1; } ;;
        num-stages|block-k)
          [[ "$step" == bytes || "$step" == stamps || "$step" == timed || "$step" == lockprobe ]] || { plan_refuse "$ln" "$k belongs to a bytes unit (or a stamps, timed or lockprobe unit), not $step"; return 1; } ;;   # A2
        histogram)
          [[ "$step" == timed || "$step" == bytes || "$step" == stamps ]] || { plan_refuse "$ln" "histogram belongs to a timed, bytes or stamps unit, not $step"; return 1; } ;;   # C3
        arms)
          [[ "$step" == timed || "$step" == bytes ]] || { plan_refuse "$ln" "arms belongs to a timed or bytes unit, not $step"; return 1; }
          [[ " $opts " == *" histogram="* ]] || { plan_refuse "$ln" "arms needs histogram=: R3 refuses --arms on a ladder page"; return 1; } ;;
        instr-variants)
          [[ "$step" == regcheck || "$step" == perturb || "$step" == stamps ]] || { plan_refuse "$ln" "instr-variants belongs to a regcheck, perturb or stamps unit, not $step"; return 1; } ;;
        declared-copies)
          case "$step" in
            timed) ;;
            bytes|floor) plan_refuse "$ln" "declared-copies belongs to a timed unit, not $step: a counter page declares R3's $COUNTER_COPIES copies (validate_counter_plan) and no other"; return 1 ;;
            *) plan_refuse "$ln" "declared-copies belongs to a timed unit, not $step"; return 1 ;;
          esac ;;
        instr-evict-a|instr-evict-b)
          [[ "$step" == bytes ]] || { plan_refuse "$ln" "$k belongs to a bytes unit, not $step (a stamps unit carries no hint)"; return 1; } ;;
        instr-stamps|instr-every|instr-marks|stamp-groups|stamp-treads|stamp-arms)
          [[ "$step" == stamps ]] || { plan_refuse "$ln" "$k belongs to a stamps unit, not $step"; return 1; } ;;
        timed-groups|timed-treads)
          [[ "$step" == timed ]] || { plan_refuse "$ln" "$k belongs to a timed unit, not $step"; return 1; } ;;
        slip-policy)
          [[ "$step" == timed ]] || { plan_refuse "$ln" "slip-policy belongs to a timed unit, not $step"; return 1; }
          [[ " $opts " == *" histogram="* || "${kv#*=}" == page ]] || { plan_refuse "$ln" "slip-policy=cell needs histogram=: R3 refuses --lock-mhz on its balanced ladder"; return 1; } ;;
        g1-rule)
          [[ "$step" == timed ]] || { plan_refuse "$ln" "g1-rule belongs to a timed unit, not $step: it is the histogram pages' G1 rule"; return 1; }
          [[ " $opts " == *" histogram="* ]] || { plan_refuse "$ln" "g1-rule needs histogram=: the rule reads a histogram page's cells.csv"; return 1; }
          [[ "${kv#*=}" == r5-any || " $opts " == *" slip-policy=cell "* ]] || { plan_refuse "$ln" "g1-rule=r7-median-run3 needs slip-policy=cell: locked_r3 takes --min-clean-repeats with --slip-policy cell only"; return 1; } ;;
        lf-arms|lf-group-m)
          [[ "$step" == launchfloor ]] || { plan_refuse "$ln" "$k belongs to a launchfloor unit, not $step"; return 1; } ;;
        block-m|block-n|num-warps)   # TILE
          [[ "$step" == timed || "$step" == bytes || "$step" == lockprobe || "$step" == stamps ]] || { plan_refuse "$ln" "$k belongs to a timed, bytes, lockprobe or stamps unit, not $step"; return 1; }   # TILE A2 C3
          for key in block-m block-n num-warps block-k num-stages; do   # TILE
            [[ " $opts " == *" $key="* ]] || { plan_refuse "$ln" "a tile is whole: $k needs block-m, block-n, num-warps, block-k and num-stages"; return 1; }   # TILE
          done ;;   # TILE
        tiles|models)   # TILE
          [[ "$step" == tilecensus ]] || { plan_refuse "$ln" "$k belongs to a tilecensus unit, not $step"; return 1; } ;;   # TILE
        lock-mhz)   # A2
          [[ "$step" == prelude || "$step" == timed || "$step" == rulers || "$step" == bytes ]] || { plan_refuse "$ln" "lock-mhz belongs to the prelude (the session's lock) or a timed, rulers or bytes unit (its own), not $step"; return 1; }   # A2 S1
          v="${kv#*=}"; [[ "$step" == prelude || ",$PROBE_LOCKS," == *",$v,"* ]] || PROBE_LOCKS="${PROBE_LOCKS:+$PROBE_LOCKS,}$v" ;;   # A2
        power-limit-w)   # A2
          [[ "$step" == prelude ]] || { plan_refuse "$ln" "power-limit-w belongs to the prelude, not $step"; return 1; } ;;   # A2
        ladder|probes)   # A2
          [[ "$step" == lockprobe ]] || { plan_refuse "$ln" "$k belongs to a lockprobe unit, not $step"; return 1; }   # A2
          v="${kv#*=}"; [[ "$k" == ladder ]] && for v in ${v//,/ }; do [[ ",$PROBE_LOCKS," == *",$v,"* ]] || PROBE_LOCKS="${PROBE_LOCKS:+$PROBE_LOCKS,}$v"; done; true ;;   # A2
        gate)   # A2
          [[ "$step" == gate ]] || { plan_refuse "$ln" "gate belongs to a gate unit, not $step"; return 1; } ;;   # A2
        unless-lock)   # A2
          [[ "$step" == timed ]] || { plan_refuse "$ln" "unless-lock belongs to a timed unit, not $step"; return 1; } ;;   # A2
        byte-treads)   # S1
          [[ "$step" == bytes ]] || { plan_refuse "$ln" "byte-treads belongs to a bytes unit, not $step"; return 1; }   # S1
          [[ " $opts " != *" histogram="* ]] || { plan_refuse "$ln" "byte-treads is a ladder page's: a histogram page's treads are its cells'"; return 1; } ;;   # S1
        replay|stall-metrics)   # S1
          [[ "$step" == bytes ]] || { plan_refuse "$ln" "$k belongs to a bytes unit, not $step"; return 1; }   # S1
          [[ " $opts " == *" histogram="* ]] || { plan_refuse "$ln" "$k needs histogram=: S1 runs it on registered cells only"; return 1; } ;;   # S1
        floor-lock-mhz|floor-base|floor-shape-metrics|floor-null-kernel)
          [[ "$step" == floor ]] || { plan_refuse "$ln" "$k belongs to a floor unit, not $step"; return 1; }
          [[ " $opts " == *" floor-groups="* ]] || { plan_refuse "$ln" "$k needs a registered floor design (floor-groups=...)"; return 1; }
          if [[ "$k" == floor-lock-mhz ]]; then
            v="${kv#*=}"
            [[ "$v" == "$LOCK_TIMED" || ",$PROBE_LOCKS," == *",$v,"* ]] || PROBE_LOCKS="${PROBE_LOCKS:+$PROBE_LOCKS,}$v"
          fi ;;
      esac
    done
    case "$step" in
      prelude|launchfloor|rulers|gpubench) ;;
      powerproof) [[ -n "$label" ]] || { plan_refuse "$ln" "a powerproof unit names its directory: label=..."; return 1; } ;;   # S1
      lockprobe) [[ -n "$label" ]] || { plan_refuse "$ln" "a lockprobe unit names its pages: label=..."; return 1; }   # A2
        for key in ladder probes block-m block-n block-k num-warps num-stages; do   # A2
          [[ " $opts " == *" $key="* ]] || { plan_refuse "$ln" "a lockprobe unit needs $key=..."; return 1; }   # A2
        done ;;   # A2
      gate) [[ -n "$label" && " $opts " == *" gate="* ]] || { plan_refuse "$ln" "a gate unit needs label= and gate="; return 1; } ;;   # A2
      tilecensus) [[ -n "$label" ]] || { plan_refuse "$ln" "a tilecensus unit names its file: label=..."; return 1; }   # TILE
        [[ " $opts " == *" tiles="* && " $opts " == *" models="* ]] || { plan_refuse "$ln" "a tilecensus unit needs tiles= and models="; return 1; } ;;   # TILE
      regcheck)
        (( seen_timed )) && { plan_refuse "$ln" "regcheck runs before any timed or deep unit (owner decision 2, 2026-10-07)"; return 1; }
        seen_regcheck=1 ;;
      perturb) seen_perturb=1 ;;
      stamps)
        [[ -n "$label" ]] || { plan_refuse "$ln" "a stamps unit names its variant: label=..."; return 1; }
        if [[ " $opts " == *" instr-variants="* ]]; then
          [[ " $opts " == *" instr-stamps="* ]] && { plan_refuse "$ln" "instr-variants and instr-stamps: one or the other"; return 1; }
          (( seen_regcheck )) || { plan_refuse "$ln" "a stamps unit with instr-variants needs a regcheck unit before it"; return 1; }
          for key in stamp-groups stamp-treads stamp-arms; do
            [[ " $opts " == *" $key="* ]] || { plan_refuse "$ln" "a stamps unit needs $key=..."; return 1; }
          done
        elif [[ " $opts " == *" histogram="* ]]; then   # C3
          for key in instr-stamps stamp-groups stamp-arms block-m block-n block-k num-warps num-stages; do   # C3
            [[ " $opts " == *" $key="* ]] || { plan_refuse "$ln" "a histogram stamps unit needs $key=..."; return 1; }   # C3
          done   # C3
          [[ " $opts " != *" stamp-treads="* ]] || { plan_refuse "$ln" "stamp-treads on a histogram stamps unit: its treads are its cells'"; return 1; }   # C3
          (( seen_regcheck )) || { plan_refuse "$ln" "a histogram stamps unit needs a regcheck unit before it"; return 1; }   # C3
        else
          for key in instr-stamps stamp-groups stamp-treads stamp-arms; do
            [[ " $opts " == *" $key="* ]] || { plan_refuse "$ln" "a stamps unit needs $key=..."; return 1; }
          done
        fi
        (( seen_perturb )) || { plan_refuse "$ln" "an instrumented unit needs a perturb unit before it (its gate)"; return 1; }
        instr_after=1 ;;
      bytes|floor) [[ -n "$label" ]] || { plan_refuse "$ln" "a $step unit names its pages' directory: label=..."; return 1; }
        if [[ " $opts " == *" instr-evict-"* ]]; then
          (( seen_perturb )) || { plan_refuse "$ln" "an instrumented unit needs a perturb unit before it (its gate)"; return 1; }
          instr_after=1
        fi ;;
      calibrate|timed|deep) [[ -n "$label" ]] || { plan_refuse "$ln" "a $step unit names its run directory: label=..."; return 1; }
        if [[ "$step" == timed && " $opts " == *" histogram="* && " $opts " != *" declared-copies="* ]]; then
          plan_refuse "$ln" "a histogram timed unit needs declared-copies=N (R3 refuses a histogram page without it)"; return 1
        fi
        [[ "$step" == calibrate ]] || seen_timed=1 ;;
      *) plan_refuse "$ln" "step $step writes into shared run directories; a plan runs prelude, bytes, floor, launchfloor, calibrate, timed and deep (rental 4 adds rulers, gpubench, perturb and stamps; rental 5 regcheck)"; return 1 ;;
    esac
    if [[ -n "$nodrop" ]]; then
      [[ "$step" != prelude ]] || { plan_refuse "$ln" "nodrop on the prelude: the prelude is never dropped"; return 1; }
      [[ -z "$group$depends" ]] || { plan_refuse "$ln" "nodrop with drop-group= or depends=: a unit the deadline never drops cannot go with a group"; return 1; }
    fi
    key="$model|$label|$step"
    [[ "$seen" == *"<$key>"* ]] && { plan_refuse "$ln" "$model $step label=$label twice: two units would write one directory"; return 1; }
    seen+="<$key>"
    if [[ -z "$est" || -z "$cap" ]]; then
      local keep="$MODEL"; MODEL="$model"
      [[ -n "$est" ]] || est="$(step_est "$step")"
      [[ -n "$cap" ]] || cap="$(step_cap "$step")"
      MODEL="$keep"
    fi
    (( cap >= est )) || { plan_refuse "$ln" "cap $cap is under the estimate $est"; return 1; }
    U_MODEL+=("$model"); U_STEP+=("$step"); U_OPTS+=("${opts# }"); U_EST+=("$est"); U_CAP+=("$cap"); U_LABEL+=("$label"); U_GROUP+=("$group"); U_DEPENDS+=("$depends"); U_NODROP+=("$nodrop")
    [[ -n "$group" ]] && groups_seen+="$group "
  done < "$f"
  (( ${#U_STEP[@]} >= 2 )) || { echo "--plan: $f holds no unit after the prelude" >&2; return 1; }
  if (( seen_perturb && ! instr_after )); then
    echo "--plan: $f has a perturb unit and no instrumented unit after it to gate" >&2; return 1
  fi
  if (( seen_regcheck && ! instr_after )); then
    echo "--plan: $f has a regcheck unit and no instrumented unit to check" >&2; return 1
  fi
}
#: Unit i's model and overrides into this shell and the environment its step inherits;
#: every override not on the line is cleared, so nothing leaks from the unit before.
apply_unit() {   # INDEX
  local i="$1" kv k v
  MODEL="${U_MODEL[$i]}"; [[ "$MODEL" == - ]] && MODEL="$DEFAULT_MODEL"
  FLOOR_GROUPS=""; FLOOR_TREADS=""; BYTE_GROUPS=""; OUT_LABEL="${U_LABEL[$i]}"
  LF_TREADS=1,2,3,4,5,6,7,8,9; LF_MODES=E240,E0,E480,GR; LF_TRACE=1,2
  LF_ARMS=native,shared,private; LF_GROUP_M=4; TIMED_GROUPS=""; TIMED_TREADS=""
  NUM_STAGES=""; BLOCK_K=""; SLOT_PAD_ROWS=""; PARTITION_METRICS=""
  FLOOR_LOCK_MHZ=""; FLOOR_BASE=1; FLOOR_SHAPE=""; FLOOR_NULL=""
  DECLARED_COPIES=""; INSTR_EVICT_A=""; INSTR_EVICT_B=""; INSTR_STAMPS=""; INSTR_EVERY=""; INSTR_MARKS=""
  STAMP_GROUPS=""; STAMP_TREADS=""; STAMP_ARMS=""; VARIANT_ID=""
  HISTOGRAM=""; R3_ARMS=""; INSTR_VARIANTS=""; VARIANT_IDS=""; SLIP_POLICY=""; G1_RULE=""
  TILE_BLOCK_M=""; TILE_BLOCK_N=""; TILE_NUM_WARPS=""; CENSUS_TILES=""; CENSUS_MODELS=""   # TILE
  GATE_NAME=""; LADDER=""; PROBES=""; UNLESS_LOCK=""; local unit_lock="" a2_lock="" a2_pw=""   # A2
  BYTE_TREADS=""; S1_REPLAY=""; S1_STALL=""   # S1
  for kv in ${U_OPTS[$i]}; do
    k="${kv%%=*}"; v="${kv#*=}"
    case "$k" in
      byte-groups) BYTE_GROUPS="$v" ;; floor-groups) FLOOR_GROUPS="$v" ;; floor-treads) FLOOR_TREADS="$v" ;;
      lf-treads) LF_TREADS="$v" ;; lf-modes) LF_MODES="$v" ;; lf-trace) LF_TRACE="$v" ;;
      lf-arms) LF_ARMS="$v" ;; lf-group-m) LF_GROUP_M="$v" ;;
      timed-groups) TIMED_GROUPS="$v" ;; timed-treads) TIMED_TREADS="$v" ;;
      num-stages) NUM_STAGES="$v" ;; block-k) BLOCK_K="$v" ;; slot-pad-rows) SLOT_PAD_ROWS="$v" ;;
      partition-metrics) [[ "$v" == 1 ]] && PARTITION_METRICS=1 ;;
      floor-lock-mhz) FLOOR_LOCK_MHZ="$v" ;; floor-base) FLOOR_BASE="$v" ;;
      floor-shape-metrics) [[ "$v" == 1 ]] && FLOOR_SHAPE=1 ;;
      floor-null-kernel) [[ "$v" == 1 ]] && FLOOR_NULL=1 ;;
      declared-copies) DECLARED_COPIES="$v" ;;
      instr-evict-a) INSTR_EVICT_A="$v" ;; instr-evict-b) INSTR_EVICT_B="$v" ;;
      instr-stamps) INSTR_STAMPS="$v" ;; instr-every) INSTR_EVERY="$v" ;; instr-marks) INSTR_MARKS="$v" ;;
      stamp-groups) STAMP_GROUPS="$v" ;; stamp-treads) STAMP_TREADS="$v" ;; stamp-arms) STAMP_ARMS="$v" ;;
      histogram) HISTOGRAM="$v" ;; arms) R3_ARMS="$v" ;; instr-variants) INSTR_VARIANTS="$v" ;;
      slip-policy) SLIP_POLICY="$v" ;;
      g1-rule) G1_RULE="$v" ;;
      block-m) TILE_BLOCK_M="$v" ;; block-n) TILE_BLOCK_N="$v" ;; num-warps) TILE_NUM_WARPS="$v" ;;   # TILE
      tiles) CENSUS_TILES="$v" ;; models) CENSUS_MODELS="$v" ;;   # TILE
      gate) GATE_NAME="$v" ;; ladder) LADDER="$v" ;; probes) PROBES="$v" ;; power-limit-w) a2_pw="$v" ;; unless-lock) UNLESS_LOCK="$v" ;;   # A2
      lock-mhz) if [[ "${U_STEP[$i]}" == prelude ]]; then a2_lock="$v"; else unit_lock="$v"; fi ;;   # A2
      byte-treads) BYTE_TREADS="$v" ;; replay) S1_REPLAY="$v" ;; stall-metrics) S1_STALL=1 ;;   # S1
    esac
  done
  if [[ "${U_STEP[$i]}" == stamps || -n "$INSTR_EVICT_A$INSTR_EVICT_B" ]]; then
    VARIANT_ID="u$(( i + 1 ))-${U_LABEL[$i]}"
    if [[ "${U_STEP[$i]}" == stamps && -n "$INSTR_VARIANTS" ]]; then
      local vv; for vv in ${INSTR_VARIANTS//,/ }; do VARIANT_IDS+="${VARIANT_IDS:+ }$VARIANT_ID-$vv"; done
      VARIANT_ID="${VARIANT_IDS%% *}"
    fi
  fi
  export MOE_DRIVER_NUM_STAGES="$NUM_STAGES" MOE_DRIVER_BLOCK_K="$BLOCK_K" MOE_DRIVER_SLOT_PAD_ROWS="$SLOT_PAD_ROWS"
  export MOE_DRIVER_PARTITION_METRICS="$PARTITION_METRICS" MOE_DRIVER_FLOOR_LOCK_MHZ="$FLOOR_LOCK_MHZ"
  export MOE_DRIVER_FLOOR_BASE="$FLOOR_BASE" MOE_DRIVER_FLOOR_SHAPE="$FLOOR_SHAPE" MOE_DRIVER_FLOOR_NULL="$FLOOR_NULL"
  export MOE_DRIVER_MODEL="$MODEL" MOE_DRIVER_FLOOR_GROUPS="$FLOOR_GROUPS" MOE_DRIVER_FLOOR_TREADS="$FLOOR_TREADS"
  export MOE_DRIVER_BYTE_GROUPS="$BYTE_GROUPS" MOE_DRIVER_OUT_LABEL="$OUT_LABEL"
  export MOE_DRIVER_LF_TREADS="$LF_TREADS" MOE_DRIVER_LF_MODES="$LF_MODES" MOE_DRIVER_LF_TRACE="$LF_TRACE"
  export MOE_DRIVER_LF_ARMS="$LF_ARMS" MOE_DRIVER_LF_GROUP_M="$LF_GROUP_M"
  export MOE_DRIVER_TIMED_GROUPS="$TIMED_GROUPS" MOE_DRIVER_TIMED_TREADS="$TIMED_TREADS"
  export MOE_DRIVER_DECLARED_COPIES="$DECLARED_COPIES" MOE_DRIVER_VARIANT_ID="$VARIANT_ID"
  export MOE_DRIVER_INSTR_EVICT_A="$INSTR_EVICT_A" MOE_DRIVER_INSTR_EVICT_B="$INSTR_EVICT_B"
  export MOE_DRIVER_INSTR_STAMPS="$INSTR_STAMPS" MOE_DRIVER_INSTR_EVERY="$INSTR_EVERY" MOE_DRIVER_INSTR_MARKS="$INSTR_MARKS"
  export MOE_DRIVER_STAMP_GROUPS="$STAMP_GROUPS" MOE_DRIVER_STAMP_TREADS="$STAMP_TREADS" MOE_DRIVER_STAMP_ARMS="$STAMP_ARMS"
  export MOE_DRIVER_HISTOGRAM="$HISTOGRAM" MOE_DRIVER_ARMS="$R3_ARMS" MOE_DRIVER_SLIP_POLICY="$SLIP_POLICY"
  export MOE_DRIVER_G1_RULE="$G1_RULE"
  export MOE_DRIVER_BLOCK_M="$TILE_BLOCK_M" MOE_DRIVER_BLOCK_N="$TILE_BLOCK_N" MOE_DRIVER_NUM_WARPS="$TILE_NUM_WARPS"   # TILE
  export MOE_DRIVER_CENSUS_TILES="$CENSUS_TILES" MOE_DRIVER_CENSUS_MODELS="$CENSUS_MODELS"   # TILE
  export MOE_DRIVER_GATE="$GATE_NAME" MOE_DRIVER_LADDER="$LADDER" MOE_DRIVER_PROBES="$PROBES" MOE_DRIVER_UNIT_LOCK="$unit_lock"   # A2
  export MOE_DRIVER_UNLESS_LOCK="$UNLESS_LOCK"   # A2
  export MOE_DRIVER_BYTE_TREADS="$BYTE_TREADS" MOE_DRIVER_S1_REPLAY="$S1_REPLAY" MOE_DRIVER_S1_STALL="$S1_STALL"   # S1
  if [[ "${U_STEP[$i]}" == prelude ]]; then export MOE_DRIVER_A2_LOCK="$a2_lock" MOE_DRIVER_POWER_LIMIT_W="${a2_pw:-$POWER_LIMIT_W}"; fi   # A2
  export MOE_DRIVER_INSTR_VARIANTS="$INSTR_VARIANTS" MOE_DRIVER_VARIANT_IDS="$VARIANT_IDS"
}
#: The plan's instrumented units as instr_probe.py's variants (JSON): every stamps unit and
#: every bytes unit with an eviction hint, in file order, ids u<unit>-<label>.
variants_json() {
  local i keep="$MODEL" first=1 g t a spec kind
  printf '['
  for (( i = 0; i < ${#U_STEP[@]}; i++ )); do
    apply_unit "$i"
    [[ -n "$VARIANT_ID" ]] || continue
    if [[ "${U_STEP[$i]}" == stamps && -n "$INSTR_VARIANTS" ]]; then
      local vid
      for vid in $VARIANT_IDS; do
        (( first )) || printf ','
        first=0
        printf '\n {"id": "%s", "kind": "%s", "model": "%s", "groups": [%s], "treads": [%s], "arms": ["%s"], "block_k": %s, "num_stages": %s, "spec": "%s"}' \
          "$vid" stamps "$MODEL" "$STAMP_GROUPS" "$STAMP_TREADS" "${STAMP_ARMS//,/\", \"}" "${BLOCK_K:-64}" "${NUM_STAGES:-4}" "$(r5_variant_spec "${vid##*-}")"
      done
      continue
    fi
    if [[ "${U_STEP[$i]}" == stamps && -n "$HISTOGRAM" ]]; then   # C3
      (( first )) || printf ','   # C3
      first=0   # C3
      printf '\n {"id": "%s", "kind": "stamps", "model": "%s", "groups": [%s], "arms": ["%s"], "block_m": %s, "block_n": %s, "block_k": %s, "num_warps": %s, "num_stages": %s, "histogram": "%s", "spec": "%s"}' $(: C3) \
        "$VARIANT_ID" "$MODEL" "$STAMP_GROUPS" "${STAMP_ARMS//,/\", \"}" "$TILE_BLOCK_M" "$TILE_BLOCK_N" "$BLOCK_K" "$TILE_NUM_WARPS" "$NUM_STAGES" "$HISTOGRAM" "$(stamp_spec)"   # C3
      continue   # C3
    fi   # C3
    if [[ "${U_STEP[$i]}" == stamps ]]; then
      kind=stamps; g="$STAMP_GROUPS"; t="$STAMP_TREADS"; a="$STAMP_ARMS"; spec="$(stamp_spec)"
    else
      kind=bytes; g="${BYTE_GROUPS:-${BYTE_GS[*]}}"; g="${g// /,}"; t="$TREADS"; a="native,shared,private"; spec="$(evict_spec)"
    fi
    (( first )) || printf ','
    first=0
    printf '\n {"id": "%s", "kind": "%s", "model": "%s", "groups": [%s], "treads": [%s], "arms": ["%s"], "block_k": %s, "num_stages": %s, "spec": "%s"}' \
      "$VARIANT_ID" "$kind" "$MODEL" "$g" "$t" "${a//,/\", \"}" "${BLOCK_K:-64}" "${NUM_STAGES:-4}" "$spec"
  done
  printf '\n]\n'
  MODEL="$keep"
}
unit_text() { local i="$1"; echo "unit $(( i + 1 ))/${#U_STEP[@]} ${U_MODEL[$i]}${U_LABEL[$i]:+ ${U_LABEL[$i]}}${U_OPTS[$i]:+ ${U_OPTS[$i]}}"; }
unit_dropped() { [[ -f "$D/plan-drops" ]] && grep -qx -- "$1" "$D/plan-drops"; }
#: A2 (re-review MINOR 5): a unit whose unless-lock is the session's lock takes 0 min and is not budgeted   # A2
a2_ruled_out() { local l; l="$(sed -n 's/^A2_LOCK=//p' "$D/a2-lock.env" 2>/dev/null)"; [[ -n "$l" && " ${U_OPTS[$1]} " == *" unless-lock=$l "* ]]; }   # A2
#: Before unit i: while the units still to run overrun the time left, drop the
#: LAST one not yet dropped (never the prelude, never a nodrop=1 unit): the file's
#: order is the priority. With only nodrop units left over the deadline none is
#: dropped (rental 6): execute_step still skips one whose own estimate no longer fits.
unit_budget() {   # INDEX
  local i="$1" j need left
  while :; do
    need=0
    for (( j = i; j < ${#U_STEP[@]}; j++ )); do unit_dropped "$j" || a2_ruled_out "$j" || need=$(( need + U_EST[j] )); done   # A2
    left="$(left_min)"
    (( need <= left )) && return 0
    for (( j = ${#U_STEP[@]} - 1; j >= i; j-- )); do
      [[ "${U_STEP[$j]}" == prelude ]] && continue
      [[ -n "${U_NODROP[$j]:-}" ]] && continue
      unit_dropped "$j" && continue
      echo "$j" >> "$D/plan-drops"
      ledger "DROPPED $(unit_text "$j") ${U_STEP[$j]}: ${need} min of units left, ${left} min to the deadline"
      # a drop-group goes together: a base page without its knob page is wasted minutes
      local k
      if [[ -n "${U_GROUP[$j]}" ]]; then
        for (( k = i; k < ${#U_STEP[@]}; k++ )); do
          [[ "$k" != "$j" && "${U_GROUP[$k]}" == "${U_GROUP[$j]}" ]] || continue
          [[ -n "${U_NODROP[$k]:-}" ]] && continue
          unit_dropped "$k" && continue
          echo "$k" >> "$D/plan-drops"
          ledger "DROPPED $(unit_text "$k") ${U_STEP[$k]}: drop-group ${U_GROUP[$j]} goes with unit $(( j + 1 ))"
        done
        drop_dependents "$i" "${U_GROUP[$j]}"
      fi
      continue 2
    done
    if ! grep -qx -- "$i" "$D/plan-nodrop-short" 2>/dev/null; then
      echo "$i" >> "$D/plan-nodrop-short"
      ledger "NODROP: ${need} min of units left against ${left} min to the deadline, and only nodrop=1 units (or the running one) are left to drop: they are not dropped: each runs while its own estimate still fits the time left (else it is SKIPPED, as any step), under its cap and the deadline"
    fi
    return 0
  done
}
#: rental 5: drop every unit not yet run whose depends= names GROUP (and theirs, in turn)
drop_dependents() {   # FROM GROUP
  local i="$1" grp="$2" k
  for (( k = i; k < ${#U_STEP[@]}; k++ )); do
    [[ "${U_DEPENDS[$k]:-}" == "$grp" ]] || continue
    [[ -n "${U_NODROP[$k]:-}" ]] && continue
    unit_dropped "$k" && continue
    echo "$k" >> "$D/plan-drops"
    ledger "DROPPED $(unit_text "$k") ${U_STEP[$k]}: it depends on drop-group $grp"
    [[ -n "${U_GROUP[$k]}" ]] && drop_dependents "$i" "${U_GROUP[$k]}"
  done
  return 0
}
#: The plan's own page directory for unit i, as the step will name it (date aside).
unit_dir() {
  local i="$1" m="${U_MODEL[$i]}"; [[ "$m" == - ]] && m="$DEFAULT_MODEL"
  case "${U_STEP[$i]}" in
    bytes|floor) echo "\$RESULTS_ROOT/<date>-<card>-$m-${U_LABEL[$i]}-r3-counters" ;;
    launchfloor) echo "\$RESULTS_ROOT/<date>-<card>-launch-floor${U_LABEL[$i]:+-${U_LABEL[$i]}}/$m" ;;
    calibrate) echo "card-wide ruler moe/bench/hardware/measured_<card>.yaml, \$D/ruler.env" ;;
    timed|deep) echo "\$RESULTS_ROOT/gaps-<card>-$m-${U_LABEL[$i]}" ;;
    rulers|gpubench|perturb|regcheck|powerproof) echo "\$RESULTS_ROOT/<date>-<card>-${U_STEP[$i]}${U_LABEL[$i]:+-${U_LABEL[$i]}}" ;;   # S1
    stamps) echo "\$RESULTS_ROOT/<date>-<card>-instr-$m-${U_LABEL[$i]}" ;;
    tilecensus) echo "\$RESULTS_ROOT/<date>-<card>-tile-census-$m-${U_LABEL[$i]}" ;;   # TILE
    lockprobe) echo "\$RESULTS_ROOT/gaps-<card>-<model>-lp<rung><letter> per rung, \$D/gate-probe-<rung>.json" ;;   # A2
    gate) echo "\$D/gate-<name>-<lock>.json (STOP on FAIL)" ;;   # A2
    *) echo "-" ;;
  esac
}
print_units() {
  local i total=0 keep="$MODEL"
  echo "THE GH200 MODEL-TEST SESSION (docs/LAMBDA.md section 3c), unattended: a PLAN of ${#U_STEP[@]} units"
  echo "  plan file: $PLAN_PATH (sha256 $(sha256_of "$PLAN_PATH"))"
  printf '  %-3s %-24s %-11s %4s %4s  %s\n' "#" model step est cap "overrides; pages"
  for (( i = 0; i < ${#U_STEP[@]}; i++ )); do
    apply_unit "$i"
    printf '  %-3s %-24s %-11s %4s %4s  %s\n' "$(( i + 1 ))" "${U_MODEL[$i]}" "${U_STEP[$i]}" \
      "${U_EST[$i]}" "${U_CAP[$i]}" "${U_LABEL[$i]:+label=${U_LABEL[$i]} }${U_OPTS[$i]:-}${U_OPTS[$i]:+; }$(unit_dir "$i")${U_GROUP[$i]:+ [drop-group ${U_GROUP[$i]}]}${U_DEPENDS[$i]:+ [depends ${U_DEPENDS[$i]}]}${U_NODROP[$i]:+ [nodrop]}"
    echo "        $(c2_what "${U_STEP[$i]}")"   # S1 C2
    total=$(( total + U_EST[i] ))
  done
  MODEL="$keep"
  echo "  $total min of units, estimated, plus setup_vm.sh when --setup"
  echo "  deadline: $([[ -n "$DEADLINE" ]] && utc_of "$DEADLINE" || echo 'none given (the run itself needs --deadline)')"
  echo "  drop order when time runs short: the last unit first, up the file (never the prelude, never a nodrop unit):"
  local order=""; for (( i = ${#U_STEP[@]} - 1; i >= 1; i-- )); do [[ -n "${U_NODROP[$i]:-}" ]] || order+=" $(( i + 1 ))"; done
  echo "   ${order}"
  echo "  pushes: after every unit and every byte page, to the run's branch (scripts/vm_results_push.sh)$( (( NO_PUSH )) && echo ': OFF (--no-push)')"
  [[ -z "$PROBE_LOCKS" ]] || echo "  the prelude also probes the floor lock(s) ${PROBE_LOCKS//,/ } MHz (lock, read back, reset)"
  local ids=""
  for (( i = 0; i < ${#U_STEP[@]}; i++ )); do
    apply_unit "$i"; [[ -n "$VARIANT_IDS" ]] && ids+=" ${VARIANT_IDS// /|}" || { [[ -n "$VARIANT_ID" ]] && ids+=" $VARIANT_ID"; }
  done
  MODEL="$keep"
  [[ -z "$ids" ]] || echo "  instrumented units, each run only on its perturb unit's hard legs and regcheck (PERTURBED on a timing FAIL alone):$ids"   # C3
}

usage() { sed -n '2,9p' "$SELF" | sed 's/^# \{0,1\}//'; }

# ---- arguments -----------------------------------------------------------
PRINT_VARIANTS=0; DRY=0; NO_PUSH=0; SETUP=0; COMMIT=""; REPO_URL=""; DEADLINE=""; FROM=""; ONLY=""; ONE_STEP=""; PLAN_FILE=""
ARGS=("$@")
while (( $# )); do
  case "$1" in
    --dry-run)  DRY=1; shift ;;
    --print-variants) PRINT_VARIANTS=1; DRY=1; shift ;;
    --no-push)  NO_PUSH=1; shift ;;
    --setup)    SETUP=1; shift ;;
    --commit)   COMMIT="${2:-}"; shift 2 ;;
    --repo)     REPO_URL="${2:-}"; shift 2 ;;
    --deadline) DEADLINE="${2:-}"; shift 2 ;;
    --from)     FROM="${2:-}"; shift 2 ;;
    --steps)    ONLY="${2:-}"; shift 2 ;;
    --model)    MODEL="${2:-}"; MODEL_GIVEN=1; shift 2 ;;
    --floor-groups) FLOOR_GROUPS="${2:-}"; shift 2 ;;
    --floor-treads) FLOOR_TREADS="${2:-}"; shift 2 ;;
    --byte-groups) BYTE_GROUPS="${2:-}"; shift 2 ;;
    --plan)     PLAN_FILE="${2:-}"; shift 2 ;;
    --step)     ONE_STEP="${2:-}"; shift 2 ;;
    -h|--help)  usage; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit "$EXIT_REFUSED" ;;
  esac
done

is_step() { local s; for s in "${STEPS[@]}"; do [[ "$s" == "$1" ]] && return 0; done; return 1; }
opt_in() { local s; for s in "${OPT_IN_STEPS[@]}"; do [[ "$s" == "$1" ]] && return 0; done; return 1; }

[[ "$MODEL" =~ ^[a-z0-9][a-z0-9.-]*$ ]] || { echo "--model $MODEL: not a model name" >&2; exit "$EXIT_REFUSED"; }
export MOE_DRIVER_MODEL="$MODEL"
for v in "$FLOOR_GROUPS" "$FLOOR_TREADS" "$BYTE_GROUPS"; do
  [[ -z "$v" || "$v" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "--floor-groups/--floor-treads/--byte-groups $v: a comma-separated list of integers" >&2; exit "$EXIT_REFUSED"; }
done
export MOE_DRIVER_FLOOR_GROUPS="$FLOOR_GROUPS" MOE_DRIVER_FLOOR_TREADS="$FLOOR_TREADS"
export MOE_DRIVER_BYTE_GROUPS="$BYTE_GROUPS" MOE_DRIVER_OUT_LABEL="$OUT_LABEL"
export MOE_DRIVER_LF_TREADS="$LF_TREADS" MOE_DRIVER_LF_MODES="$LF_MODES" MOE_DRIVER_LF_TRACE="$LF_TRACE"
R3_BASE[1]="$MODEL"; R1_BASE[1]="$MODEL"
#: One declaration for the whole session (2026-09-28). Mixtral's auto rule
#: declares 9 copies at every ladder depth; at E = 64 it would declare 6, 8 or
#: 9 by the page's treads, and the byte model refuses timed pages whose
#: declaration is not the counter pages' (the counter ladder's 9). A model
#: other than Mixtral's pins the counter pages' count on every timed page.
#: Rental 4: a timed unit's declared-copies key replaces it (and reaches Mixtral too).
if [[ -n "$DECLARED_COPIES" ]]; then
  R3_BASE+=(--declared-copies "$DECLARED_COPIES")
else
  case "$MODEL" in mixtral-*) ;; *) R3_BASE+=(--declared-copies "$COUNTER_COPIES") ;; esac
fi

# One step, in its own process: the orchestrator runs this under a cap.
if [[ -n "$ONE_STEP" ]]; then
  is_step "$ONE_STEP" || { echo "no step $ONE_STEP (${STEPS[*]})"; exit "$EXIT_REFUSED"; }
  enter "$ONE_STEP"
  CURRENT="$ONE_STEP"
  if [[ -n "$UNLESS_LOCK" && "${A2_LOCK:-}" == "$UNLESS_LOCK" ]]; then   # A2
    ledger "$ONE_STEP SKIPPED BY RULE: the session's lock is $A2_LOCK MHz and this unit runs only off it (unless-lock=$UNLESS_LOCK)"; exit 0   # A2
  fi   # A2
  echo "== $(utc) step $ONE_STEP: $(step_what "$ONE_STEP")"
  "step_$ONE_STEP"
  exit $?
fi

PLAN=()
if [[ -n "$ONLY" ]]; then
  for s in ${ONLY//,/ }; do is_step "$s" || { echo "no step $s (${STEPS[*]})" >&2; exit "$EXIT_REFUSED"; }; done
  for s in "${STEPS[@]}"; do [[ ",$ONLY," == *",$s,"* ]] && PLAN+=("$s"); done
elif [[ -n "$FROM" ]]; then
  is_step "$FROM" || { echo "no step $FROM (${STEPS[*]})" >&2; exit "$EXIT_REFUSED"; }
  on=0; for s in "${STEPS[@]}"; do [[ "$s" == "$FROM" ]] && on=1; (( on )) && ! opt_in "$s" && PLAN+=("$s"); done
elif [[ "$MODEL" != "$DEFAULT_MODEL" ]]; then
  # P1's locks and R1 answered this card's clock question on 8x7B; another
  # model's session measures its bytes, times and floor (docs/registered)
  for s in "${STEPS[@]}"; do [[ "$s" == eta || "$s" == r1lock ]] || opt_in "$s" || PLAN+=("$s"); done
else
  for s in "${STEPS[@]}"; do opt_in "$s" || PLAN+=("$s"); done
fi

if [[ -n "$PLAN_FILE" ]]; then
  for given in "$ONLY" "$FROM" "$FLOOR_GROUPS" "$FLOOR_TREADS" "$BYTE_GROUPS" "${MODEL_GIVEN:-}"; do
    [[ -z "$given" ]] || { echo "--plan carries every unit's model, steps and overrides: drop --model, --steps, --from, --floor-*, --byte-groups" >&2; exit "$EXIT_REFUSED"; }
  done
  PLAN_PATH="$(resolve_plan "$PLAN_FILE")" || { echo "--plan $PLAN_FILE: no such file (looked in \$MOE_HOME/repo, as given, and beside this driver)" >&2; exit "$EXIT_REFUSED"; }
  load_plan "$PLAN_PATH" || exit "$EXIT_REFUSED"
  # the prelude probes every floor lock the plan names besides 1710
  export MOE_DRIVER_PROBE_LOCKS="$PROBE_LOCKS"
fi

if (( PRINT_VARIANTS )) && [[ -n "$PLAN_FILE" ]]; then
  # rental 4: the plan's instrumented units as instr_probe.py reads them; nothing runs
  variants_json
  exit "$EXIT_REFUSED"
fi
if (( DRY )) && [[ -n "$PLAN_FILE" ]]; then
  # A2 (build review fix 7): on a box with a GPU, the card's power.max_limit is read before the plan prints, so a
  # A2: a card under the prelude's power-limit-w stops vm_run.sh start before setup (no plan header, no setup)
  if command -v nvidia-smi >/dev/null 2>&1; then   # A2
    a2_pw="$(grep -oE 'power-limit-w=[0-9]+' <<< "${U_OPTS[0]:-}" | cut -d= -f2)"; a2_pw="${a2_pw:-$POWER_LIMIT_W}"   # A2
    a2_max="$(nvidia-smi --query-gpu=power.max_limit --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"   # A2
    if [[ ! "$a2_max" =~ ^[0-9]+([.][0-9]+)?$ ]]; then   # A2 (re-review MINOR 2): no numeric reading, the prelude decides
      echo "power.max_limit not read pre-setup ('${a2_max:0:60}'): the prelude's P-W check decides after setup" >&2   # A2
    elif ! awk -v p="$a2_max" -v w="$a2_pw" 'BEGIN { exit !(p + 0 >= w - 0.5) }'; then   # A2
      echo "REFUSED (pre-setup, tileA2-gates P-W): power.max_limit reads '$a2_max' W, under the plan's $a2_pw W: no setup, no plan"   # A2
      exit "$EXIT_REFUSED"   # A2
    fi   # A2
  fi   # A2
  print_units
  echo "DRY RUN: nothing ran, nothing was written (exit 2, as every dry run in this repo)"
  exit "$EXIT_REFUSED"
fi
if (( DRY )); then
  print_plan
  echo "DRY RUN: nothing ran, nothing was written (exit 2, as every dry run in this repo)"
  exit "$EXIT_REFUSED"
fi

[[ "$DEADLINE" =~ ^[0-9]+$ ]] || { echo "--deadline <epoch seconds> is required (the laptop sets launch + 400 min)" >&2; exit "$EXIT_REFUSED"; }
command -v timeout >/dev/null || { echo "no timeout(1) on PATH: each step runs under a cap" >&2; exit "$EXIT_REFUSED"; }

# ---- setup: setup_vm.sh, then the checkout's own copy of this driver -------
if (( SETUP )); then
  [[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "--setup needs --commit <full sha>" >&2; exit "$EXIT_REFUSED"; }
  [[ -n "$REPO_URL" ]] || { echo "--setup needs --repo <url>" >&2; exit "$EXIT_REFUSED"; }
  D="$MOE_HOME/session/gh200-driver"; mkdir -p "$D"
  SV="$(dirname "$SELF")/setup_vm.sh"
  [[ -f "$SV" ]] || { ledger "setup REFUSED: no $SV (scripts/vm_run.sh copies it beside this driver)"; exit "$EXIT_REFUSED"; }
  ledger "setup_vm.sh start: commit $COMMIT from $REPO_URL, deadline $(utc_of "$DEADLINE")"
  rc=0
  timeout -k 60 3600 bash "$SV" --repo "$REPO_URL" --commit "$COMMIT" > "$HOME/setup_vm.log" 2>&1 < /dev/null || rc=$?
  ledger "setup_vm.sh exit $rc ($(grep -cE '^RESULT: [A-Z]+ PF[0-9] PASS' "$HOME/setup_vm.log" 2>/dev/null || echo 0) preflight checks passed)"
  if (( rc != 0 )); then
    push_results "setup_vm.sh exit $rc: STOPPED before any measurement" "$rc"
    exit "$rc"
  fi
  push_results "setup_vm.sh: READY"
  NEXT="$MOE_HOME/repo/scripts/$(basename "$SELF")"   # TILE
  [[ -f "$NEXT" ]] || { ledger "setup: no $NEXT after setup"; exit "$EXIT_ERROR"; }
  [[ "$(sha256_of "$NEXT")" == "$(sha256_of "$SELF")" ]] \
    || ledger "setup: the checkout's driver differs from the copy that ran setup; the checkout's runs"
  REST=()
  i=0
  while (( i < ${#ARGS[@]} )); do
    case "${ARGS[$i]}" in
      --setup) i=$(( i + 1 )) ;;
      --commit|--repo) i=$(( i + 2 )) ;;
      *) REST+=("${ARGS[$i]}"); i=$(( i + 1 )) ;;
    esac
  done
  exec bash "$NEXT" ${REST[@]+"${REST[@]}"}
fi

# ---- the run ---------------------------------------------------------------
[[ -f "$ENV_SH" ]] || { echo "no $ENV_SH: run setup_vm.sh first (or --setup)" >&2; exit "$EXIT_REFUSED"; }
# shellcheck disable=SC1090
. "$ENV_SH"
S="$SESSION_ROOT"; D="$S/gh200-driver"
mkdir -p "$S/logs" "$D"
# rental 4: the plan's instrumented units, for the perturb and stamps steps
[[ -n "$PLAN_FILE" ]] && { variants_json > "$D/instr-variants.json"; apply_unit 0; }
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
trap 'on_signal HUP' HUP
ledger "driver start: commit $(git -C "$REPO" rev-parse HEAD 2>/dev/null), steps $([[ -n "$PLAN_FILE" ]] && echo "of the plan $PLAN_FILE" || echo "${PLAN[*]}"), deadline $(utc_of "$DEADLINE") ($(left_min) min of work left)"
#: One step, its estimate already settled: skip it when it no longer fits, else
#: run it under its cap, record it, make the card safe and push. Stops the whole
#: session (exit) on what stops it; returns otherwise. CAP_MIN empty: step_cap's.
execute_step() {   # NAME EST [CAP_MIN] [UNIT_TEXT]
  CURRENT="$1"; local est="$2" capm="${3:-}" utext="${4:-}" left why cap t0 rc mins what
  local sname="$CURRENT${utext:+ ($utext)}"
  left="$(left_min)"
  if (( est <= 0 || est > left )); then
    if (( est <= 0 )); then why="every part of it is dropped"
    else why="its estimate, $est min, is over the $left min left"; fi
    ledger "SKIPPED $sname: $why"
    printf '%-10s SKIPPED (%s)%s\n' "$CURRENT" "$why" "${utext:+ [$utext]}" >> "$D/summary.new"
    (( FINAL < EXIT_INVALID )) && FINAL=$EXIT_INVALID
    if [[ "$CURRENT" == prelude ]]; then
      ledger "STOPPED: the prelude was skipped, and every step after it needs it"
      cp "$D/summary.new" "$D/summary.txt"
      push_results "prelude: SKIPPED, the session STOPPED" "$FINAL"
      exit "$FINAL"
    fi
    if [[ "$CURRENT" == gate || "$CURRENT" == lockprobe ]]; then   # A2
      ledger "STOPPED: $sname was skipped, and a skipped gate never passes (tileA2-gates)"   # A2
      cp "$D/summary.new" "$D/summary.txt"   # A2
      push_results "$sname: SKIPPED, the session STOPPED (tileA2-gates)" "$FINAL"   # A2
      exit "$FINAL"   # A2
    fi   # A2
    push_results "$sname: SKIPPED ($why)"
    return 0
  fi
  cap=$(( ${capm:-$(step_cap "$CURRENT")} * 60 ))
  (( cap > left * 60 )) && cap=$(( left * 60 ))
  ledger "$CURRENT START: $(c2_what "$CURRENT")$(: S1 C2)${utext:+ [$utext]} (estimate $est min, cap $(( cap / 60 )) min, $left min left)"
  t0=$(now_s)
  run_step "$CURRENT" "$cap"; rc=$?
  mins=$(( ( $(now_s) - t0 + 30 ) / 60 ))
  case $rc in
    124|137) what="STOPPED at its cap" ;;
    *) what="exit $rc" ;;
  esac
  ledger "$CURRENT END: $what after $mins min${utext:+ [$utext]}"
  printf '%-10s %s after %s min%s\n' "$CURRENT" "$what" "$mins" "${utext:+ [$utext]}" >> "$D/summary.new"
  if [[ "$CURRENT" == prelude && $rc -ne 0 ]]; then
    ledger "STOPPED: the prelude did not pass, and every step after it needs it"
    FINAL=$EXIT_REFUSED
    hygiene || true
    push_results "prelude: exit $rc, the session STOPPED" "$FINAL"
    cp "$D/summary.new" "$D/summary.txt"
    exit "$FINAL"
  fi
  case $rc in
    0|1) ;;
    2|3|124) (( FINAL < EXIT_INVALID )) && FINAL=$EXIT_INVALID ;;
    *) FINAL=$EXIT_ERROR ;;
  esac
  if ! hygiene; then
    ledger "STOPPED after $sname: the card could not be made safe for the next step"
    FINAL=$EXIT_ERROR
    cp "$D/summary.new" "$D/summary.txt"
    push_results "$sname: $what; hygiene FAILED, the session STOPPED" "$FINAL"
    exit "$FINAL"
  fi
  if [[ ( "$CURRENT" == gate || "$CURRENT" == lockprobe ) && $rc -ne 0 ]]; then   # A2
    ledger "STOPPED: $sname read FAIL ($what): the registered early abort ends the rental (tileA2-gates)"   # A2
    FINAL=$EXIT_INVALID   # A2
    cp "$D/summary.new" "$D/summary.txt"   # A2
    push_results "$sname: GATE FAILED, the session STOPPED (tileA2-gates)" "$FINAL"   # A2
    exit "$FINAL"   # A2
  fi   # A2
  cp "$D/summary.new" "$D/summary.txt"
  push_results "$sname: $what ($mins min)"
}

FINAL=$EXIT_DONE
: > "$D/summary.new"
if [[ -n "$PLAN_FILE" ]]; then
  ledger "plan: $PLAN_PATH (sha256 $(sha256_of "$PLAN_PATH")), ${#U_STEP[@]} units"
  for (( i = 0; i < ${#U_STEP[@]}; i++ )); do
    unit_budget "$i"
    if unit_dropped "$i"; then
      apply_unit "$i"
      execute_step "${U_STEP[$i]}" 0 "" "$(unit_text "$i")"
      continue
    fi
    apply_unit "$i"
    execute_step "${U_STEP[$i]}" "${U_EST[$i]}" "${U_CAP[$i]}" "$(unit_text "$i")"
  done
else
  for (( i = 0; i < ${#PLAN[@]}; i++ )); do
    budget "$i"
    execute_step "${PLAN[$i]}" "$(est_now "${PLAN[$i]}")"
  done
fi
CURRENT=""
ledger "driver end: exit $FINAL"
cp "$D/summary.new" "$D/summary.txt"
push_results "the session ended: driver exit $FINAL" "$FINAL"
exit "$FINAL"
