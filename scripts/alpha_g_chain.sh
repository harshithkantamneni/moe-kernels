#!/bin/bash
# The alpha(G) matrix session: the chain that produces the shared-over-private
# ratio table the analytical model needs, with both ratio arms held OFF the
# power cap, every geometry's clock elasticity measured beside it (the tables
# say per G whether a ratio reads as a re-read fraction at all), and every G
# run as a seed triple scored jointly.
#
#   bash scripts/alpha_g_chain.sh --dry-run          # plan and price, nothing measured
#   bash scripts/alpha_g_chain.sh                    # a new chain session on this card
#   bash scripts/alpha_g_chain.sh --new              # a new one beside an existing one, on purpose
#   bash scripts/alpha_g_chain.sh --resume           # continue the newest one holding CHAIN.tsv
#   SESSION=<dir> bash scripts/alpha_g_chain.sh      # continue a named one
#   bash scripts/alpha_g_chain.sh --resume --past-gpu-tests   # go on past a red test_gpu.py, recorded
#   bash scripts/alpha_g_chain.sh --resume --past-v8          # go on past the probe check or the pilot's V8, recorded
#   END_SUITE=run bash scripts/alpha_g_chain.sh --resume      # buy the end suite too (off by default)
# AN OVERRIDE HOLDS. Its decision is written to the ledger as its own
# OVERRIDDEN row, and every later pass of the session reads that row back: a
# plain --resume after it goes on past the same gate without the flag, and
# says so on the console.
#
# ON THE POD, FIRST. The volume's clone carries the last session's ruler yaml
# modified, and git will not switch branches over it, even when it is
# byte-identical to the committed one:
#   git -C /workspace/moe-kernels checkout -- moe/bench/hardware/measured_nvidia_h200.yaml
#   git -C /workspace/moe-kernels fetch origin
#   git -C /workspace/moe-kernels checkout -B r3-align origin/r3-align
#   git -C /workspace/moe-kernels log -1 --format=%h   # must print the head that was pushed
# Then launch it DETACHED, so a dropped ssh session does not take a run of
# four hours and more with it, and watch the log and the ledger. The console
# APPENDS (>>): a --resume launched the same way keeps the earlier passes'
# lines, and the exfil line printed at the end carries that file:
#   cd /workspace/moe-kernels && nohup setsid bash scripts/alpha_g_chain.sh \
#       >> /workspace/alpha_g_chain.out 2>&1 < /dev/null &
#   tail -f /workspace/alpha_g_chain.out       # and $SESSION/CHAIN.tsv, one row per step
# The chain's calibrate re-dirties that yaml: --publish writes this card's
# ruler into the tracked file, every row after it carries git_dirty, and the
# yaml is committed WITH the results. The exfil line carries it and
# calibrate's own run directory.
#
# WHICH SESSION. A bare run REFUSES when a chain session for this card already
# holds CHAIN.tsv: every R1 and R3 run id carries the session's tag, so a
# fresh tag re-measures every arm that session finished. --resume takes the
# newest directory holding CHAIN.tsv, never a dry run's (CHAIN-dryrun.tsv
# only), and --new opens a fresh one on purpose. --dry-run always plans into a
# fresh directory of its own, and with --resume or SESSION= it is REFUSED: it
# would overwrite that session's step logs, and the driver's thermal.log, with
# plan pages. A measuring run is REFUSED on a box whose card cannot be named
# (the probe's reason is printed); it holds a lock on the session for the
# whole run (a second chain on it is refused; a held flock, the pod's, is never
# taken over, since its holder may be an arm a killed chain left running, and
# the refusal names fuser, lsof and a ps line that needs neither; only the
# mkdir fallback's --resume takes over a lock whose pid is dead or whose host
# differs); and on its first pass it records the card's UUID in
# $SESSION/DEVICE and R3's duty in $SESSION/R3_DUTY, refusing a resume on
# another card or at another R3_DUTY (--new opens a session at another duty).
# The DEVICE check refuses before any step runs; R3's DEVICE file and R1's
# device_guard each refuse a directory measured on another card as well, so a
# replacement pod's card is refused by either layer alone.
#
# WHAT IT RUNS, IN ORDER, and why that order:
#   preflight      the two arms' --self-test on the box's interpreter: the
#                  scorers are proven before a cent is spent. Each must be
#                  DONE; an INVALID self-test is a scorer that failed its own
#                  planted world, and it STOPS the chain. One that is not DONE
#                  runs again on every pass (seconds on the CPU), so a
#                  --resume after the scorer is fixed re-proves it.
#   preconditions  scripts/h200_gaps_session.sh --only thermal,calibrate,pin_probe-n64-g1
#                  in THIS session directory: card healthy, ruler measured on
#                  this card. The chain STOPS when the driver exits 2 (its
#                  thermal, calibration or reference-grade gate refused), and
#                  unless thermal and calibrate are DONE in its ARMS.tsv. The
#                  step latches only on the driver's exit 0, so a row the
#                  driver still owes is re-attempted on --resume.
#                  pin_probe-n64-g1 RUNS FOR THE RECORD AND IS NOT GATED: it
#                  asks whether MOE_FORCE_TILE reaches the kernel, and neither
#                  arm below reads MOE_FORCE_TILE. Both pin through vLLM's
#                  override_config.
#   counter-probe  INFORMATIONAL: NEVER GATED AND NEVER LATCHED. Can this pod
#                  read a DRAM counter: it decides whether the one route to
#                  alpha at G >= 2 at this tile (ncu bytes, section 7 of
#                  session 5's findings) can happen on RunPod at all. It
#                  finds ncu on PATH, then at NCU_SEARCH's globs
#                  (/usr/local/cuda*/bin/ncu and
#                  /opt/nvidia/nsight-compute/*/ncu by default), and runs
#                  scripts/dram_counter_route.py --probe, the driver's
#                  counter_plan probe, from PY_BASE with the first one's
#                  directory first on PATH: one real kernel under ncu, and
#                  dram__bytes_read.sum read back or refused. Its row's state
#                  is INFO; the verdict (OPEN, BLOCKED, ABSENT, UNTESTED or
#                  ERROR), ncu's path and version, the exact error and the two
#                  capabilities go into the note and $SESSION/COUNTERS, beside
#                  the probe's own payload, $SESSION/COUNTERS.json. It runs on
#                  every measuring pass (a counter route is a property of the
#                  pod), priced in minutes at the driver's own arm_minutes for
#                  counter_plan and capped like an arm; a dry run prices it and
#                  writes a SKIPPED row, and times nothing. RunPod's record,
#                  as this repo commits it: two rented H200s attempted a
#                  counter read and both were refused with ERR_NVGPUCTRPERM,
#                  on 2026-08-25 (ncu over the harness's own CLI:
#                  profiles/q2_kernel_names.txt) and on 2026-09-15, on a pod
#                  holding neither CAP_SYS_ADMIN nor CAP_PERFMON; the
#                  2026-09-09 and 2026-09-10 pods were never asked (their
#                  probe profiled /bin/true); in session 4 (2026-09-21) ncu was
#                  absent from the image as far as the probe looked, which was
#                  PATH alone (REFUSE, "no ncu on PATH": results/published/
#                  2026-09-21-nvidia_h200-session4/session/
#                  gaps-nvidia_h200-20260921T235000Z/counter_route.json, on
#                  pod-h200-session4); session 5 attempted none. Two refused
#                  pods are a record, not a fact about the platform: the probe
#                  answers for the pod it runs on.
#   gpu-tests      tests/test_gpu.py on this card, from PY_BASE (the venv
#                  WITHOUT vLLM): the timing and clock primitives both arms
#                  stand on, on the card that will time them, in minutes. It
#                  does NOT cover the graph probe R3's V8 stands on: that one
#                  test imports vLLM and SKIPS from PY_BASE. Not green STOPS
#                  the chain, and an exit 0 in which no test passed is not
#                  green (off a card every one of them skips);
#                  --past-gpu-tests goes on and writes that decision to the
#                  ledger.
#   probe-check    scripts/private_weight_reference.py --probe-check from
#                  PY_VLLM: that skipped test's on-card check alone
#                  (moe_align_block_size captured under a CUDA graph, the
#                  calls per replay as registered, not host-bound, the graph's
#                  per-call time under the eager p50), about a minute, before
#                  the pilot spends a ten-minute ladder finding the same
#                  thing. Not DONE STOPS the chain and names the page. It runs
#                  again on every pass until it is DONE. After INVALID,
#                  UNKNOWN or ERROR the remedy is a code change
#                  (PROBE_CALLS_PER_REPLAY, or the capture) and a --resume
#                  after it must re-prove the probe; --past-v8, the same
#                  instrument as the pilot's V8, goes on and writes that
#                  decision to the ledger. REFUSED timed nothing (no card, no
#                  vLLM, or vLLM's op did not import): the interpreter or the
#                  card is wrong, the arms run from the same one, and the STOP
#                  names PY_VLLM and a --resume after that fix, not --past-v8.
#   r3-g<G>-s0     scripts/private_weight_reference.py at --duty 0.25, every G
#                  at seed 0 first, so the ratio arm's never-run paths (the
#                  duty timer, the probe inside a page) meet the card minutes
#                  in, not an hour in. At 0.25 session 4's clock arm sat flat
#                  at the ceiling; at 0.5 it still tracked board power. The
#                  chain STOPS after r3-g1-s0 when its V8 is not PASS, and when
#                  it wrote no report.json: V8 describes the instrument, not G,
#                  and the probe re-runs on every page, so a V8 UNKNOWN or FAIL
#                  makes every later ratio page INVALID. --past-v8 goes on and
#                  writes that decision to the ledger; it buys R1's regime
#                  words and ratio pages that cannot be quoted (the STOP prices
#                  both off the arms' own plans), and SEEDS=0 limits those
#                  pages to seed 0.
#   r1-g<G>        scripts/clock_elasticity.py at each G, three duty states
#                  (1.0 0.5 0.25: the card on its power cap at 1.0, off it at
#                  0.5 and 0.25; R1_DUTY says why these), the per-M-tile
#                  elasticity gated: the REGIME word for that G, below.
#   r3-g<G>-s1,s2  seeds 1 then 2 at every G, each later seed scored WITH the
#                  earlier ones of its G through --replicate-of (DESIGN
#                  DECISION 14). The R1 block sits between seed 0 and seed 1:
#                  a G's seed 0 and seed 1 are the rest of the seed-0 block and
#                  the whole R1 block apart, its seed 1 and seed 2 one seed
#                  block apart, and the dry run prints both spacings off the
#                  arms' own prices. A G whose seed-0 page did not read V7 PASS
#                  gets a SKIPPED row (not latched) for seeds 1 and 2, which
#                  are not run, and the note says which of two things
#                  happened. V7 FAIL: the two ratio arms ran at different
#                  clocks at this duty (the power state), and a later seed at
#                  it would buy the same split. A V7 FAIL at 0.25 means that
#                  duty is not yet flat for that arm on this card; the page
#                  names a lower duty; the chain skips the G's later seeds and
#                  prints the follow-up command: that G's three seeds at
#                  --duty 0.1, by hand AFTER the chain and never beside it,
#                  priced off R3's own plan at 0.1, a new design key that is
#                  read with --read on the laptop, outside PAIRS.tsv. V7
#                  UNKNOWN, absent or unreadable: seed 0's V7 could not be
#                  scored (nothing timed, e.g. the sweep was skipped on V8, or
#                  a clock was unread), so a later seed would read the same;
#                  the seed-0 log says which.
#   suite          OFF BY DEFAULT. END_SUITE=run buys the whole suite,
#                  uncapped, from PY_BASE, AFTER every arm: a record of this
#                  box that gates nothing, its row in the ledger. Without it
#                  (END_SUITE=skip, the default) the step writes a SKIPPED
#                  row saying the suite was not requested. Why off: on
#                  session 5's pod the base-venv suite exercised nothing the
#                  arms depend on beyond tests/test_gpu.py, which runs above,
#                  and it ran ~1.4 s a test off the network volume (2499 tests
#                  in 3472 s before it was interrupted at 49%). A
#                  suite that already ran to its tally, green or red (pytest
#                  exit 1), is not bought again and gets no row over it; a
#                  timeout, an interrupted run or a log with no tally re-runs.
#                  Both pytest steps run with this chain's own knobs (SESSION,
#                  END_SUITE, G_LADDER and the rest of CHAIN_KNOBS) removed
#                  from their environment: the suite's tests spawn this chain.
#
# EVERY ARM STEP (probe-check, r1, r3) runs under `timeout --signal=INT
# --kill-after=60`, the pytest steps' wrapper, at a cap off its own price:
# max(3 x the arm's own --dry-run figure, 30 min), the plan run again just
# before the step, and for probe-check, which prints no plan, the same rule
# over its documented constant. The counter probe runs under the same rule
# over its price, the driver's booking for counter_plan, and a timeout there is
# an INFO row whose note reads ERROR: TIMED OUT. A timed-out arm is ERROR with
# TIMED OUT in its note, not latched: both arms resume per cell, so --resume
# re-runs it. A box without timeout runs them uncapped, as it runs the pytest
# steps. A cap rescues a stall a signal can reach; a process parked inside the
# volume's FUSE request cannot be signalled, by this or by anything.
#
# THE REGIME WORD, per G, off R1's interval through the arm's own band_of, and
# what the ratio beside it reads as, which both tables print as `reads_as` (the
# rule H200 session 5's findings support):
#   RAW-STANDS        wholly below 0.25: the per-tile cost is a traffic
#                     quantity, and the ratio beside it is a re-read fraction.
#                     The one word that reads `re-read fraction`.
#   UNREGISTERED-GAP  wholly inside [0.25, 0.40]: NEITHER registered
#                     consequence is licensed; the ratio is quoted with the
#                     interval and no word. Reads `unresolved`.
#   CLOCK-CARRIES     wholly above 0.40: the ratio is NOT alpha. Session 5 at
#                     G >= 4: the shared arm sits on a per-tile floor that
#                     scales with the SM clock, any alpha in [0, 0.60] fits it
#                     equally, and the bytes-rate bound (below) still proves
#                     real reuse. Reads `blend (traffic and a clock-scaled
#                     on-chip floor): not alpha`.
#   STRADDLES         the interval crosses an edge: no word. Reads `unresolved`.
#   withheld:<EXIT>   R1's page exited INVALID, REFUSED, ERROR or unscored: no
#                     word is read off a page its own gates did not stand behind.
#                     Reads `unresolved`.
#   unmeasured        no R1 report for that G on disk yet. Reads `unresolved`.
# R1'S RESOLUTION. Session 4's G=16 claim over treads 2 and deeper read a
# half-width of 0.084 over its states 1.0, 0.5 and 0.25 (0.092 over all four;
# the all-tread reading's was 0.076) against R1's 0.075 target, half the gap
# band's width. Session 5 ran R1 at those three states, the chain's, at every
# G and read half-widths of 0.0096 at G=1 (STRADDLES at 0.40), 0.0660 at G=4
# (CLOCK-CARRIES), 0.0837 then 0.0909 at G=16 (withheld:INVALID both times,
# on V5 and then on V7) and 0.0875 at G=64 (CLOCK-CARRIES): its pages
# 0d8858eb, e1c429b7, a5a8fde2.first-v5-invalid then a5a8fde2, and a3d5cd3a
# under results/published/2026-09-23-nvidia_h200-session5/results/
# gaps-nvidia_h200/clock_elasticity/. An interval within its half-width of
# 0.25 or 0.40 reads STRADDLES.
# THE WORD IS A SECANT, between the capped clock at duty 1.0 and the clocks at
# 0.5 and 0.25; R3 runs at 0.25, at the ceiling, the top of that secant's
# range. To first order, for a per-tile cost A + B/f with A and B not
# negative, the local elasticity B/(Af + B) lies in [0, 1] and falls as f
# rises, so RAW-STANDS carries over to R3's operating point and
# CLOCK-CARRIES is only an upper bound there. That form cannot produce an
# elasticity above 1, which session 4's G=16 claim read at every subset of
# its states: where R1's point is above 1 the A + B/f reading does not
# apply, and the interval is quoted without it.
#
# WHAT IT LEAVES: $SESSION/CHAIN.tsv (one row per step: state, rc, seconds,
# log, note, with the driver's second opinion taken off the RESULT lines);
# the tables, REBUILT from the reports on disk at the end of every pass and
# before every STOP, so a resume never duplicates a row and an R1 finished on
# a later pass is joined, with their legend in $SESSION/PAIRS-README.txt:
# $SESSION/PAIRS.tsv, one row per ratio run: G, seed, ratio, interval, exit
# word and its scope (`alone`, or `envelope` when C1 was scored with earlier
# seeds), duty, run id; the joint reading on that run's page (n, spread, sd,
# envelope, verdict), only where the run formed a ratio and is in it; each
# arm's median clock over the ladder and the count of LEVEL LOW (arm, tread)
# cells; R1's eta, interval, word and exit; and `reads_as`, what the ratio can
# be read as off that word (THE REGIME WORD, above). $SESSION/PAIRS-by-G.tsv,
# one row per G, every seed of it that formed a ratio read together by R3's
# own cross-run machinery whatever order they ran in (n, mean, sd, envelope,
# joint verdict, any INVALID run inside the envelope) beside R1's columns and
# `reads_as`: the per-G value. It also carries THE BYTES-RATE BOUND, the one
# bound on alpha that needs no private arm (session 5's findings, 3.6):
# alpha <= (t x C / W - 1) / (n - 1), t the shared arm's time at its top tread
# n off the reports' own ladders (the mean over the G's runs), W the expert
# set off their memory plans, and C the ruler's read_stream and its pin rate,
# off the yaml calibrate wrote in this session (the tracked one after exfil),
# held to the bandwidth the reports were scored against; each rebuild prints
# it per G with the ceiling it used. $SESSION/PAIRS-fixed.tsv holds the
# coordinates every row shares (model, tile, pinned config, treads, repeats,
# duty) and where each was read, and the bound's inputs. $SESSION/COUNTERS and
# COUNTERS.json, the counter probe's verdict and payload. $SESSION/DEVICE and
# $SESSION/R3_DUTY, logs under $SESSION/chain-logs/, the driver's own ARMS.tsv
# beside them, and a follow-up's commands, when a V7 FAIL named one, in
# $SESSION/followup-g<G>.txt. --resume skips every step whose newest row is
# latched (DONE, CLAIM_FAIL, INVALID) and re-runs REFUSED, ERROR, UNKNOWN and
# SKIPPED ones, with four exceptions: a preflight self-test and the probe
# check run again until they are DONE, the counter probe runs on every pass
# (its INFO row is never latched), and the end suite is not bought again once
# it ran to its tally.
#
# THE THREE HABITS THIS REPOSITORY HAS BEEN BURNED BY, and how this file
# avoids them: no `set -e` (a failed arm is a ledger row, not the end of a
# rented session); every measured command's exit code is captured with
# `|| rc=$?` and never through a pipeline; nothing here starts a background
# job, so there is no PID variable.
set -uo pipefail

# >>> LIFTABLE
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY_BASE="${PY_BASE:-/workspace/venvs/base/bin/python}"
PY_VLLM="${PY_VLLM:-/workspace/venvs/vllm/bin/python}"
[[ -x "$PY_BASE" ]] || PY_BASE="$REPO/.venv/bin/python"
[[ -x "$PY_BASE" ]] || PY_BASE="$(command -v python3 || true)"
[[ -x "$PY_VLLM" ]] || PY_VLLM="$PY_BASE"
HELPERS="$REPO/scripts/alpha_g_chain_helpers.py"

#: The ladder. 1, 16 and 64 are the swizzles vLLM's tuned config for this model
#: on this card actually uses; 4 is the mid-doubling. G=32 is not booked: with
#: at most 8 x 6 real M-tiles per launch it is the same ordering as 64 at every
#: tread but two. Three seeds so a spread is a spread and not a range.
G_LADDER="${G_LADDER:-1 4 16 64}"
SEEDS="${SEEDS:-0 1 2}"
#: The ratio arms off the cap (DESIGN DECISION 15), at 0.25: on session 4's
#: clock arm, duty 0.5 still tracked board power (-1.09 MHz/W over 1882-1965
#: MHz, DRIFT failing in one cell in six), and duty 0.25 sat flat at 1965 MHz
#: with no drift. R3's own --duty default stays 1.0: it is a design key, and a
#: default that moved would refuse --replicate-of against session 4's runs.
R3_DUTY="${R3_DUTY:-0.25}"
R3_TREADS=6
R3_REPEATS=9
#: The duty a V7 FAIL's follow-up names: the lowest state session 4's clock
#: arm measured (1965-1980 MHz at 0.1, R3's FLAT_DUTY_EVIDENCE). A follow-up
#: is a new design key and its own runs, by hand after the chain, read with
#: --read on the laptop and never joined into PAIRS.tsv.
R3_FOLLOWUP_DUTY=0.1
#: What a ratio run costs beyond its plan's figure, which leaves out the
#: Triton compiles and the 25.4 GB private weight build ("NOT IN THAT
#: FIGURE"): an ALLOWANCE, not a measurement, charged per run.
R3_RUN_OVERHEAD_S=60
#: The probe check prints no plan, so its price is a documented ALLOWANCE,
#: not a measurement: a cold vLLM import and a CUDA context off the network
#: volume, and one probe cell (a few hundred ms of warmup and trials, eager
#: and under the graph), put at two minutes. The volume once stalled a cold
#: import for 9 minutes and recovered; the cap covers that, not this price.
PROBE_CHECK_S=120
#: WHERE THE COUNTER PROBE LOOKS FOR ncu BEYOND PATH: globs, searched in this
#: order after PATH, each one's matches newest name first. dram_counter_route.py
#: --probe asks PATH alone, and session 4's image read "no ncu on PATH" there;
#: the CUDA toolkit's bin and Nsight Compute's own directory hold an ncu that
#: no PATH entry names. Set it to add a place an image uses.
NCU_SEARCH="${NCU_SEARCH:-/usr/local/cuda*/bin/ncu /opt/nvidia/nsight-compute/*/ncu}"
#: The exfil allowance: the tar of the session, the results and the ruler.
EXFIL_S=300
#: THE HANG CAP ON AN ARM STEP: max(ARM_CAP_FACTOR x its price, ARM_CAP_FLOOR_S),
#: and ARM_CAP_UNPRICED_S when its plan priced nothing: four times R1's ~19
#: minutes, the longest arm's price at its states 1.0 0.5 0.25 (1124 s on
#: 2026-09-23's dry run; it was ~15 minutes at 1.0 0.7 0.5). 3x is past any
#: honest overrun of a plan that already charges its idle gaps; the 30-minute
#: floor is past the 9-minute cold-import stall the volume has shown and then
#: recovered from.
ARM_CAP_FACTOR=3
ARM_CAP_FLOOR_S=1800
ARM_CAP_UNPRICED_S=4500
#: The elasticity arm's states, 1.0 0.5 0.25 (the owner's decision in session
#: 5, 2026-09-23); three states is MIN_STATES. Session 4 had read 0.5, 0.25
#: and 0.10 as one clock cluster, so the chain took 1.0 0.7 0.5. On session
#: 5's pod (H200, 700 W cap) G=1's R1 at 1.0 0.7 0.5 excluded 22.4% of its
#: rows for in-burst clock drift (0.7: 30.8%, 0.5: 36.5%, 1.0: 0%) and failed
#: V4, which holds the excluded rows to 20%. At 1.0 0.5 0.25 every G passed
#: V4, excluding 16.3% (G=1), 6.7% (G=4), 5.4% and 5.8% (G=16, first page
#: and re-run) and 5.8% (G=64), and every page passed V1: the states
#: separated in clock at every tread. Median board power per state there:
#: 694-695 W at 1.0 (on the cap), 485-506 W at 0.5, 300-318 W at 0.25.
R1_DUTY="${R1_DUTY:-1.0 0.5 0.25}"
R1_TREADS=8
R1_REPEATS=13
RATE_USD_H="${RATE_USD_H:-4.59}"
#: The price per collected test ON A POD, not on a laptop: session 5's end
#: suite (pod w226zpjmj8p1d3, 1x H200, 2026-09-23;
#: results/published/2026-09-23-nvidia_h200-session5/session/
#: alpha_g-nvidia_h200-20260923T163248Z/chain-logs/suite.log) ran 2499 tests
#: in 3472 s from the base venv off the network volume (10 failed, 2470
#: passed, 19 skipped in 3471.97 s) before it was interrupted at 49%: 1.39 s
#: a test. The old 0.66 was session 4's capped run, 2242 tests in 1465 s. A
#: dry run multiplies the rate by each pytest step's collected count;
#: tests/test_gpu.py is priced at it too.
SUITE_RATE_WHO="session 5's pod rate"
[[ -n "${SUITE_S_PER_TEST:-}" ]] && SUITE_RATE_WHO="the rate SUITE_S_PER_TEST sets"
SUITE_S_PER_TEST="${SUITE_S_PER_TEST:-1.39}"
#: A hung suite (the volume's MooseFS has hung before) must not eat the
#: booking: about 1.7x the priced run at the rate above (a dry run with
#: END_SUITE=run prints it: 6946 s for the 4997 tests of 2026-09-23's tree),
#: then pytest is interrupted and its tally of what ran is still printed.
#: test_gpu.py gets its own, smaller cap.
SUITE_TIMEOUT_S="${SUITE_TIMEOUT_S:-12000}"
GPU_TESTS_TIMEOUT_S="${GPU_TESTS_TIMEOUT_S:-900}"
#: THE END SUITE IS OFF BY DEFAULT (the owner's decision in session 5,
#: 2026-09-23): `skip` records it as a SKIPPED row saying it was not
#: requested, and `run` buys it after every arm. On session 5's pod the
#: base-venv suite exercised nothing the arms depend on beyond
#: tests/test_gpu.py, which the chain runs before them either way, and it
#: ran 2499 tests in 3472 s before it was interrupted at 49%. Any other value
#: is REFUSED before a session is opened: with `skip` the default, a
#: mistyped `run` would otherwise skip the suite it asked for. A suite that
#: already ran to its tally gets no row at all (end_suite_recorded).
END_SUITE="${END_SUITE:-skip}"
#: The seed a G starts at, and the step after which V8 is read.
FIRST_SEED="${SEEDS%% *}"
FIRST_G="${G_LADDER%% *}"
PILOT="r3-g$FIRST_G-s$FIRST_SEED"

#: exit_codes.py's table, mirrored for the ledger word; compared with the
#: module by the driver's tests and by this file's.
state_word() {
  case "$1" in
    0) echo DONE ;; 1) echo CLAIM_FAIL ;; 2) echo REFUSED ;; 3) echo INVALID ;;
    4) echo ERROR ;; *) echo UNKNOWN ;;
  esac
}

#: The SECOND OPINION, the driver's rule: a page's RESULT lines must support
#: the integer the process returned, or the row is UNKNOWN and not latched.
#: $1 rc, $2 what the log implies (a code, NONE, or UNREADABLE). Prints
#: STATE<TAB>note.
state_for() {
  local rc="$1" implied="$2" word
  word="$(state_word "$rc")"
  case "$implied" in
    UNREADABLE*) printf 'UNKNOWN\tSECOND OPINION UNAVAILABLE: exit %s, the log could not be classified\n' "$rc"; return 0 ;;
    NONE)
      case "$rc" in
        0) printf 'UNKNOWN\tUNEARNED DONE: exit 0 with no RESULT line; not latched\n' ;;
        2) printf 'REFUSED\t\n' ;;
        3) printf 'UNKNOWN\tUNEARNED INVALID: exit 3 with no RESULT line; not latched\n' ;;
        4) printf 'ERROR\tcrash before any gate; re-runs on --resume\n' ;;
        *) printf 'UNKNOWN\texit %s with no RESULT line; not latched\n' "$rc" ;;
      esac
      return 0 ;;
  esac
  if [[ "$implied" == "$rc" ]]; then
    printf '%s\tlog agrees: RESULT lines imply %s\n' "$word" "$implied"
  else
    printf 'UNKNOWN\tDEFECT: page implies %s (%s), exit %s; not latched\n' \
      "$implied" "$(state_word "$implied")" "$rc"
  fi
}

#: A step's newest ledger word, or nothing when it has no row. $1 the step
#: (or the driver's arm), $2 the ledger (CHAIN.tsv or the driver's ARMS.tsv).
newest_state() {
  local name="$1" ledger="$2"
  [[ -f "$ledger" ]] || return 0
  awk -F'\t' -v n="$name" '$1==n {s=$2} END {print s}' "$ledger"
}

#: Is a step's newest ledger row a latched state: a RESULT, which --resume
#: does not buy again. NOT the same question as "is it DONE": an INVALID or a
#: CLAIM_FAIL row is latched too, so no gate asks this one.
latched() {
  case "$(newest_state "$1" "$2")" in DONE|CLAIM_FAIL|INVALID) return 0 ;; *) return 1 ;; esac
}

#: The driver's dirty count, verbatim: rc handled apart from the pipeline.
dirty_count() {
  local out
  out="$(git -C "$REPO" status --porcelain --untracked-files=all 2>/dev/null)" || { echo "-"; return 0; }
  printf '%s\n' "$out" | grep -c . || true
}

#: Run a command from the repository root: the suite's rootdir and testpaths.
in_repo() { ( cd "$REPO" && "$@" ); }

#: pytest's closing tally off a log ("4772 passed, 57 skipped in ..."), or a
#: word saying there was none. One line, no tabs: it goes in a ledger note.
suite_tally() {
  local line
  line="$(grep -E '[0-9]+ (passed|failed|errors?|skipped|tests? collected)' "$1" 2>/dev/null | tail -1)"
  line="$(printf '%s' "$line" | sed -E 's/^=+ //; s/ =+$//' | tr '\t' ' ')"
  printf '%s\n' "${line:-no tally in the log}"
}

#: How many tests a tally says passed (or were collected, $2 = collected).
tally_count() {
  local n
  n="$(printf '%s\n' "$1" | grep -oE "[0-9]+ ${2:-passed}" | grep -oE '^[0-9]+' | head -1)"
  printf '%s\n' "${n:-0}"
}

#: A --collect-only log priced at the pod's rate: prints "<count> <seconds>",
#: the seconds rounded half up (awk's n * r + 0.5, not Python's half to even).
price_tests() {
  local n
  n="$(tally_count "$(suite_tally "$1")" "tests? collected")"
  printf '%s %s\n' "$n" "$(awk -v n="$n" -v r="$SUITE_S_PER_TEST" 'BEGIN {printf "%d", n * r + 0.5}')"
}

#: Has the END SUITE left its record: its newest row that is not SKIPPED is
#: DONE, or is ERROR at pytest exit 1 with a tally (it ran to its end and some
#: tests failed; the tally and -rfE say which). It gates nothing, so a record
#: is not bought again, green or red, and END_SUITE=skip writes no row over
#: it. A timeout, an interrupted run, a crash, a REFUSED interpreter or a log
#: with no tally is no record: the next pass that runs the suite runs it.
#: Prints the record's state and note; returns 1 when there is none.
end_suite_recorded() {
  [[ -f "$LEDGER" ]] || return 1
  awk -F'\t' '$1 == "suite" && $2 != "SKIPPED" {s = $2; rc = $3; note = $7}
    END {
      if (s == "DONE" || (s == "ERROR" && rc == "1" && note !~ /no tally in the log/)) {
        print s ": " note; exit 0
      }
      exit 1
    }' "$LEDGER"
}

#: A row for a step this pass decided not to run, and why. SKIPPED is not a
#: latched word: the next pass asks again.
skip_row() {
  printf '%s\tSKIPPED\t-\t0\t%s\t-\t%s\n' "$1" "$(dirty_count)" "$2" >> "$LEDGER"
  printf '%-16s %-10s %s\n' "$1" SKIPPED "$2"
}

#: Run one step and write its row. $1 the opinion, $2 name, $3 log, $4.. the
#: command. The command's exit code is captured directly: never `cmd | tee`,
#: which reports tee's. Returns the command's rc. The opinion is an ARGUMENT,
#: never an environment variable: `OPINION=suite f` on a function exports it
#: to every child of f, and the end suite's own tests then scored their rows
#: with it. The state is decided by the opinion:
#:   page    (run_step) the arm's RESULT lines must support its exit code.
#:   page:N  (arm_step) the same, for an arm run under a cap of N seconds: an
#:           exit 124 (timeout's own code; no arm returns it), or 137 once
#:           the cap has passed (the KILL 60 s after the INT), is ERROR with
#:           TIMED OUT in the note, not latched, whatever the log printed
#:           before the cap. N 0 is no cap.
#:   driver  the preconditions step, the driver sequencing arms, which prints
#:           no RESULT line of its own: exit 0 is DONE, exit 2 is REFUSED (one
#:           of its gates refused the card or the ruler), and ANY OTHER CODE
#:           is UNKNOWN, not latched, because the driver exits 3 when a row is
#:           still owed and 4 when an arm crashed, and --resume re-attempts
#:           both only if this step runs again. Its per-arm second opinion is
#:           its own ARMS.tsv, which the chain reads back.
#:   suite   pytest, which prints no RESULT line either: exit 0 with at least
#:           one test passed is DONE; exit 0 with none passed is UNKNOWN (off
#:           a card every GPU test skips, and that is not green); any other
#:           code is ERROR. Never a latched word unless DONE, so a red
#:           tests/test_gpu.py re-runs on --resume; the end suite, which gates
#:           nothing, has its own rule (end_suite_recorded). The note is
#:           pytest's tally.
#:   collect a dry run's `pytest --collect-only`: DONE when it collected.
run_step_as() {
  local opinion="$1" name="$2" log="$3" cap=0; shift 3
  local rc=0 t0 secs implied state note tally
  case "$opinion" in page:*) cap="${opinion#page:}"; opinion=page ;; esac
  t0="$(date +%s)"
  "$@" > "$log" 2>&1 || rc=$?
  secs="$(( $(date +%s) - t0 ))"
  case "$opinion" in
    driver)
      case "$rc" in
        0) state=DONE; note="the driver's own ARMS.tsv carries each arm's second opinion" ;;
        2) state=REFUSED; note="the driver refused: its thermal, calibration or reference-grade gate spoke" ;;
        *) state=UNKNOWN
           note="driver exit $rc ($(state_word "$rc")): a row in ARMS.tsv is owed or crashed; not latched, --resume re-runs the driver" ;;
      esac ;;
    suite)
      tally="$(suite_tally "$log")"
      if (( rc == 0 )) && (( $(tally_count "$tally") > 0 )); then
        state=DONE; note="pytest exit 0: $tally"
      elif (( rc == 0 )); then
        state=UNKNOWN; note="UNEARNED DONE: pytest exit 0 but no test passed ($tally); not latched"
      elif (( rc == 124 )); then
        state=ERROR; note="pytest TIMED OUT (exit 124): $tally"
      else
        state=ERROR; note="pytest exit $rc: $tally"
      fi ;;
    collect)
      tally="$(suite_tally "$log")"
      if (( rc == 0 )) && (( $(tally_count "$tally" "tests? collected") > 0 )); then
        state=DONE
      else
        state=ERROR
      fi
      note="pytest exit $rc: $tally" ;;
    *)
      if (( cap > 0 )) && { (( rc == 124 )) || { (( rc == 137 )) && (( secs >= cap )); }; }; then
        state=ERROR
        note="TIMED OUT after $secs s against a cap of $cap s (exit $rc: INT at the cap, KILL 60 s later); not latched, --resume re-runs it and the arm resumes per cell"
      else
        implied="$("$PY_BASE" "$HELPERS" verdict "$log" 2>/dev/null)" || implied="UNREADABLE"
        IFS=$'\t' read -r state note < <(state_for "$rc" "$implied")
      fi ;;
  esac
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$state" "$rc" "$secs" \
    "$(dirty_count)" "$log" "$note" >> "$LEDGER"
  printf '%-16s %-10s rc=%s %5ss  %s\n' "$name" "$state" "$rc" "$secs" "$note"
  return "$rc"
}

#: An arm's step, its page's RESULT lines the second opinion. $1 name, $2 log,
#: $3.. the command.
run_step() { run_step_as page "$@"; }

#: An arm's step on the card, under its hang cap. $1 name, $2 log, $3 the cap
#: in seconds (`cap_for`), $4.. the command. The pytest steps' wrapper,
#: `timeout --signal=INT --kill-after=60`; a box without timeout runs the arm
#: uncapped, as it runs those.
arm_step() {
  local name="$1" log="$2" cap="$3"; shift 3
  local -a tmo=()
  if command -v timeout >/dev/null 2>&1; then
    tmo=(timeout --signal=INT --kill-after=60 "$cap")
  else
    cap=0
  fi
  run_step_as "page:$cap" "$name" "$log" ${tmo[@]+"${tmo[@]}"} "$@"
}

#: THE CAP ON ONE ARM STEP, off its price in seconds ($1, empty when its plan
#: priced nothing): max(ARM_CAP_FACTOR x the price, ARM_CAP_FLOOR_S), or
#: ARM_CAP_UNPRICED_S. Prints "<cap><TAB><how it was reached>".
cap_for() {
  local est="$1" cap
  if [[ -z "$est" ]]; then
    printf '%s\tno price on its plan: the unpriced cap\n' "$ARM_CAP_UNPRICED_S"; return 0
  fi
  cap=$(( ARM_CAP_FACTOR * est ))
  if (( cap < ARM_CAP_FLOOR_S )); then
    printf '%s\t%s x %s s is under the %s s floor\n' "$ARM_CAP_FLOOR_S" "$ARM_CAP_FACTOR" "$est" "$ARM_CAP_FLOOR_S"
  else
    printf '%s\t%s x its plan'"'"'s %s s\n' "$cap" "$ARM_CAP_FACTOR" "$est"
  fi
}

#: An arm's own price: its --dry-run, run into $1, priced off the plan page
#: it printed. $2.. the dry command. Prints the seconds, or nothing.
plan_price() {
  local log="$1"; shift
  "$@" > "$log" 2>&1 || true
  "$PY_BASE" "$HELPERS" estimate "$log" 2>/dev/null || true
}

#: The driver's own booking for one of its arms, in minutes, lifted from its
#: `arm_minutes` the way its tests lift it; nothing when it books none.
driver_minutes() {
  ( eval "$(sed -n '/^arm_minutes()/,/^esac; }/p' "$REPO/scripts/h200_gaps_session.sh")"
    arm_minutes "$1" ) 2>/dev/null
}

#: Every variable this chain reads from its environment. A pytest step runs
#: WITHOUT them: the suite's tests spawn this chain and the driver with
#: os.environ merged in, so an operator's SESSION=<dir> or END_SUITE=skip
#: reached those children and steered them into the real session.
#: MOE_RESULTS_DIR stays (tests/conftest.py sandboxes it), and so do PY_BASE
#: and PY_VLLM (tests/_hermetic.py replaces them).
CHAIN_KNOBS="REPO SESSION SESSION_ROOT RESULTS_ROOT WORKSPACE G_LADDER SEEDS R3_DUTY R1_DUTY NCU_SEARCH RATE_USD_H SUITE_S_PER_TEST SUITE_TIMEOUT_S GPU_TESTS_TIMEOUT_S END_SUITE LOCK_TOOL CAPABILITY"
without_knobs() {
  local -a unset_args=()
  local k
  for k in $CHAIN_KNOBS; do unset_args+=(-u "$k"); done
  env ${unset_args[@]+"${unset_args[@]}"} "$@"
}

#: The tests' interpreter must NOT import vLLM: the tests plant every refusal
#: door, and from a venv with vLLM an unplanted bare --run would MEASURE
#: (pod_session.sh P11c). Returns 0 when the interpreter is safe.
suite_interpreter_ok() { ! "$1" -c "import vllm" >/dev/null 2>&1; }

#: A pytest step: the interpreter refusal, the timeout wrapper, the run.
#: $1 the step's name, $2 the timeout in seconds, $3.. pytest's arguments.
pytest_step() {
  local name="$1" secs="$2"; shift 2
  local -a tmo=()
  if ! suite_interpreter_ok "$PY_BASE"; then
    printf '%s\tREFUSED\t2\t0\t%s\t-\t%s\n' "$name" "$(dirty_count)" \
      "the interpreter $PY_BASE imports vllm; an unplanted --run would MEASURE" >> "$LEDGER"
    printf '%-16s %-10s %s\n' "$name" REFUSED "$PY_BASE imports vllm; set PY_BASE to the venv without it"
    return 0
  fi
  command -v timeout >/dev/null 2>&1 && tmo=(timeout --signal=INT --kill-after=60 "$secs")
  run_step_as suite "$name" "$LOGS/$name.log" in_repo without_knobs ${tmo[@]+"${tmo[@]}"} \
    "$PY_BASE" -m pytest "$@" || true
}

#: Go on past a gate on its flag, and write that decision to the ledger as its
#: own OVERRIDDEN row, so it travels with the results. $1 the row's name, $2
#: the flag, $3 what was gone past. THE DECISION HOLDS: every later pass of
#: the session reads the row back (`overridden`) and goes on past the same
#: gate without the flag. Until 2026-09-22 a flag counted for the pass it was
#: given on, so a plain --resume after a killed chain stopped at the same gate
#: again, and re-bought tests/test_gpu.py first.
override_row() {
  printf '%s\tOVERRIDDEN\t-\t0\t%s\t-\t%s\n' "$1" "$(dirty_count)" \
    "the operator went on with $2; $3" >> "$LEDGER"
  echo "  $2: going on past $3; the ledger says so, and later passes hold to it"
}

#: Does the ledger hold an OVERRIDDEN row named $1: an override an earlier
#: pass (or this one) wrote. Returns 0 when it does.
overridden() {
  [[ -f "$LEDGER" ]] || return 1
  awk -F'\t' -v n="$1" '$1 == n && $2 == "OVERRIDDEN" {f = 1} END {exit !f}' "$LEDGER"
}

#: Go on past a gate on an override an earlier pass wrote, and say so. $1 the
#: row's name, $2 the flag, $3 what is gone past now. No new row: the decision
#: is the operator's, and it is already in the ledger.
held_override() {
  local said
  said="$(awk -F'\t' -v n="$1" '$1 == n && $2 == "OVERRIDDEN" {s = $7} END {print s}' "$LEDGER")"
  echo "  $2 holds from an earlier pass (the ledger's $1 row: $said);"
  echo "  going on past $3 without the flag"
}

#: The preflight gate: both self-tests DONE, which is not the same as latched.
#: Returns 0 to go on, 3 to stop.
preflight_gate() {
  local step st
  for step in preflight-r1 preflight-r3; do
    st="$(newest_state "$step" "$LEDGER")"
    if [[ "$st" != DONE ]]; then
      echo "STOP: $step is ${st:-absent}, not DONE; the scorer is not proven on this interpreter."
      echo "  Read $LOGS/$step.log: an INVALID or CLAIM_FAIL self-test is a scorer that"
      echo "  failed its own planted world, and every page it scores would carry that."
      echo "  Fix the scorer, bring the pod's checkout to the fix, and --resume: a"
      echo "  self-test that is not DONE runs again on every pass."
      return 3
    fi
  done
  return 0
}

#: The preconditions gate: the driver did not refuse, and thermal and calibrate
#: are DONE in its ARMS.tsv. $1 the driver's ARMS.tsv. pin_probe-n64-g1 is not
#: read: neither arm reads MOE_FORCE_TILE. Returns 0 to go on, 3 to stop.
preconditions_gate() {
  local arms="$1" need st why
  if [[ "$(newest_state preconditions "$LEDGER")" == REFUSED ]]; then
    why="$(grep -m1 '^REFUSED' "$LOGS/preconditions.log" 2>/dev/null)"
    echo "STOP: the driver REFUSED this card or its ruler (exit 2): ${why:-read the log}"
    echo "  Read $LOGS/preconditions.log. No R1 or R3 cell is scored against a ruler"
    echo "  arm 0 did not stand behind."
    return 3
  fi
  for need in thermal calibrate; do
    st="$(newest_state "$need" "$arms")"
    if [[ "$st" != DONE ]]; then
      echo "STOP: the driver's $need row is ${st:-absent}, not DONE, in $arms;"
      echo "  nothing below can be scored. Read $LOGS/preconditions.log."
      return 3
    fi
  done
  return 0
}

#: The gpu-tests gate: tests/test_gpu.py's newest row DONE, or --past-gpu-tests
#: on this pass or an earlier one. $1 1 when the flag was given. Returns 0 to
#: go on, 3 to stop.
gpu_tests_gate() {
  local past="$1" newest
  newest="$(newest_state gpu-tests "$LEDGER")"
  [[ "$newest" == DONE ]] && return 0
  if (( past )); then
    override_row gpu-tests-override --past-gpu-tests "tests/test_gpu.py's newest row: ${newest:-never ran}"
    return 0
  fi
  if overridden gpu-tests-override; then
    held_override gpu-tests-override --past-gpu-tests "tests/test_gpu.py's newest row: ${newest:-never ran}"
    return 0
  fi
  echo "STOP: tests/test_gpu.py is ${newest:-never run}, not green, on this card. Read"
  echo "  $LOGS/gpu-tests.log (-rfE lists every failure and error). If none of it bears"
  echo "  on the arms, go on with --resume --past-gpu-tests, which the ledger records"
  echo "  and every later pass holds to."
  return 3
}

#: What going on past V8 buys, priced off the arms' own plans, for the probe
#: check's STOP and the pilot's; never for a REFUSED probe check, whose arms
#: would refuse from the same interpreter. $1 how many ratio pages are already
#: on disk (0 before the pilot, 1 after it). Each price is the arm's
#: --dry-run, run again here into chain-logs.
v8_consequences() {
  local done_pages="$1" n_g=0 n_s=0 r1_s=0 est per g later seed0
  for g in $G_LADDER; do
    n_g=$(( n_g + 1 ))
    # shellcheck disable=SC2046
    est="$(plan_price "$LOGS/price-r1-g$g.log" $(r1_cmd "$g" 1))"
    r1_s=$(( r1_s + ${est:-0} ))
  done
  for g in $SEEDS; do n_s=$(( n_s + 1 )); done
  # shellcheck disable=SC2046
  est="$(plan_price "$LOGS/price-r3-g$FIRST_G.log" $(r3_cmd "$FIRST_G" "$FIRST_SEED" 1))"
  per=$(( ${est:-0} + R3_RUN_OVERHEAD_S ))
  later=$(( n_g * n_s - done_pages ))
  seed0=$(( n_g - done_pages ))
  echo "  V8 UNKNOWN or FAIL makes every later ratio page INVALID: the probe re-runs on"
  echo "  every page. --past-v8 buys R1's $n_g regime words (~$(( (r1_s + 59) / 60 )) min off R1's own"
  echo "  plans) and $later ratio pages that cannot be quoted (~$(( (later * per + 59) / 60 )) min at $per s a page,"
  echo "  R3's own plan plus ${R3_RUN_OVERHEAD_S} s of compiles and weight build; a V8 FAIL page skips its"
  echo "  sweep and costs less). SEEDS=$FIRST_SEED limits the ratio pages to seed $FIRST_SEED ($seed0, ~$(( (seed0 * per + 59) / 60 )) min):"
  echo "      SEEDS=$FIRST_SEED bash scripts/alpha_g_chain.sh --resume --past-v8"
  echo "  The ledger records --past-v8, and every later pass holds to it."
}

#: The probe-check gate: its newest row DONE, or --past-v8 on this pass or an
#: earlier one. $1 1 when the flag was given. Returns 0 to go on, 3 to stop.
#: THE STOP IS WORDED BY THE STATE. REFUSED (exit 2) timed nothing: R3
#: refuses the check with no card, no vLLM, or a vLLM op that did not import,
#: so the interpreter or the card is wrong, not the probe, and the arms run
#: from that same interpreter. Until 2026-09-22 a REFUSED check got a FAIL's
#: advice: a code change to the probe, and --past-v8 priced as if ratio pages
#: would run, which, once taken, holds for every later pass even after the
#: interpreter is fixed. INVALID, UNKNOWN and ERROR keep that advice.
probe_check_gate() {
  local past="$1" newest said
  newest="$(newest_state probe-check "$LEDGER")"
  [[ "$newest" == DONE ]] && return 0
  if (( past )); then
    override_row probe-check-override --past-v8 "the probe check's newest row: ${newest:-never ran}"
    return 0
  fi
  if overridden probe-check-override; then
    held_override probe-check-override --past-v8 "the probe check's newest row: ${newest:-never ran}"
    return 0
  fi
  said="$(grep -m1 -E '^(RESULT|REFUSED)' "$LOGS/probe-check.log" 2>/dev/null)"
  if [[ "$newest" == REFUSED ]]; then
    echo "STOP: the probe check is REFUSED, not DONE, on this card: it timed nothing."
    echo "  Read $LOGS/probe-check.log:"
    echo "      ${said:-no REFUSED line in it}"
    echo "  A refusal names what the check could not reach (a card, vLLM, or vLLM's op):"
    echo "  the interpreter or the card is wrong, not the probe. PY_VLLM here is $PY_VLLM"
    echo "  (it falls back to PY_BASE when the vLLM venv's python is missing). Check that"
    echo "  it imports vllm, and its torch wheel against the driver (torch.cuda.is_available()"
    echo "  and nvidia-smi's driver version). Fix that, then --resume, which runs this check"
    echo "  again until it is DONE. --past-v8 buys nothing here: R1 and R3 run from the same"
    echo "  interpreter, and the ledger would hold that override for every later pass."
    return 3
  fi
  echo "STOP: the probe check is ${newest:-never run}, not DONE, on this card: the"
  echo "  alignment probe under the graph did not earn PASS. Read $LOGS/probe-check.log:"
  echo "      ${said:-no RESULT or REFUSED line in it}"
  echo "  The pilot's V8 runs this probe inside R3, on every page."
  v8_consequences 0
  echo "  A fix is a code change (PROBE_CALLS_PER_REPLAY, or the capture) brought to the"
  echo "  pod's checkout, then --resume, which runs this check again until it is DONE."
  return 3
}

#: The run's report.json, off the run id its plan page printed, or nothing.
#: $1 the step's log, $2 the experiment (the ratio arm's by default).
report_of() {
  local exp="${2:-private_weight_reference}" rid
  rid="$("$PY_BASE" "$HELPERS" run-id "$1" "$exp" 2>/dev/null)" || rid=""
  [[ -n "$rid" && -f "$RESULTS/$exp/$rid/report.json" ]] && echo "$RESULTS/$exp/$rid/report.json"
  return 0
}

#: The pilot gate: the first ratio run's V8, off its report.json. $1 1 when
#: --past-v8 was given. An override on an earlier pass holds. Returns 0 to go
#: on, 3 to stop.
v8_gate() {
  local past="$1" rep v8 probe=""
  rep="$(report_of "$LOGS/$PILOT.log")"
  if [[ -z "$rep" ]]; then
    v8="unread: no report.json"
  else
    v8="$("$PY_BASE" "$HELPERS" gate "$rep" V8 2>/dev/null)" || v8="unreadable"
    probe="$("$PY_BASE" "$HELPERS" probe-note "$rep" 2>/dev/null)" || probe=""
  fi
  [[ "$v8" == PASS ]] && return 0
  if (( past )); then
    override_row v8-override --past-v8 "$PILOT's V8 $v8"
    return 0
  fi
  if overridden v8-override; then
    held_override v8-override --past-v8 "$PILOT's V8 $v8"
    return 0
  fi
  echo "STOP: $PILOT's V8 is $v8, not PASS. V8 is the alignment probe under the"
  echo "  graph: it describes the instrument, not G. Read $LOGS/$PILOT.log."
  [[ -n "$probe" ]] && echo "  the pilot's probe: $probe"
  v8_consequences 1
  return 3
}

#: Seed 0's V7 verdict for a G, or nothing when seed 0 left no report.
seed0_v7() {
  local rep
  rep="$(report_of "$LOGS/r3-g$1-s$FIRST_SEED.log")"
  [[ -n "$rep" ]] || return 0
  "$PY_BASE" "$HELPERS" gate "$rep" V7 2>/dev/null || echo unreadable
}

#: REWRITE PAIRS.tsv, PAIRS-by-G.tsv, PAIRS-fixed.tsv and PAIRS-README.txt from
#: the reports on disk. Idempotent. The helper's first line is the row count;
#: the lines after it are the ruler the bytes-rate bound is read at and one
#: line per G, the bound with the ceiling it used and what the ratio reads as.
rebuild_pairs() {
  local out rc=0
  out="$("$PY_BASE" "$HELPERS" pairs-table "$SESSION" "$RESULTS" "$G_LADDER" "$SEEDS" \
    "model=--model mixtral-8x7b" "block_m=--block-m 32" \
    "r3_treads=--treads $R3_TREADS" "r3_repeats=--repeats $R3_REPEATS" "r3_duty=--duty $R3_DUTY" \
    "r1_treads=--treads $R1_TREADS" "r1_repeats=--repeats $R1_REPEATS" "r1_duty=--duty $R1_DUTY" \
    2>&1)" || rc=$?
  if (( rc == 0 )); then
    echo "pairs     $SESSION/PAIRS.tsv (${out%%$'\n'*}, rebuilt from the reports); per G, every seed read"
    echo "          together, in PAIRS-by-G.tsv; fixed coordinates in PAIRS-fixed.tsv; the legend"
    echo "          in PAIRS-README.txt"
    [[ "$out" == *$'\n'* ]] && printf '%s\n' "${out#*$'\n'}" | sed 's/^/          /'
  else
    echo "pairs     NOT rebuilt (exit $rc): $out"
  fi
  return 0
}

#: THE COUNTER PROBE'S PRICE in seconds: the driver's own booking for the same
#: probe, its counter_plan arm (in minutes), or nothing when it books none.
counter_probe_price() {
  local m
  m="$(driver_minutes counter_plan)"
  [[ -n "$m" ]] && echo $(( m * 60 ))
  return 0
}

#: THE COUNTER PROBE, one measuring pass's. INFORMATIONAL: it gates nothing,
#: and its row's state is INFO, which no pass latches, so every pass asks
#: again (a counter route is a property of the pod). $1 its price in seconds,
#: empty when unpriced. It finds ncu (`ncu-locate`: PATH, then NCU_SEARCH),
#: runs scripts/dram_counter_route.py --probe (the driver's counter_plan
#: probe, not a second copy of it) from PY_BASE under the arm cap, with the
#: found ncu's directory first on PATH, and hands the probe's payload
#: ($SESSION/COUNTERS.json) and page to `counters`, which writes
#: $SESSION/COUNTERS and the note. Returns 0 whatever the probe read.
counter_probe_step() {
  local price="$1" log="$LOGS/counter-probe.log" located bin where cands cap how
  local rc=0 hrc=0 t0 secs out note
  local -a pathenv=() tmo=()
  located="$("$PY_BASE" "$HELPERS" ncu-locate "$NCU_SEARCH" 2>/dev/null)" || located=""
  [[ -n "$located" ]] || located=$'none\tnone\tnone'
  IFS=$'\t' read -r bin where cands <<< "$located"
  if [[ "$bin" != none && "$where" != PATH ]]; then
    pathenv=("PATH=$(dirname "$bin"):$PATH")
  fi
  IFS=$'\t' read -r cap how < <(cap_for "$price")
  if command -v timeout >/dev/null 2>&1; then
    tmo=(timeout --signal=INT --kill-after=60 "$cap")
  else
    cap=0; how="no timeout(1) on this box: uncapped"
  fi
  echo "  counter-probe: capped at $cap s ($how); informational, it gates nothing"
  rm -f "$SESSION/COUNTERS.json"
  t0="$(date +%s)"
  env ${pathenv[@]+"${pathenv[@]}"} ${tmo[@]+"${tmo[@]}"} "$PY_BASE" \
    "$REPO/scripts/dram_counter_route.py" --probe --out "$SESSION/COUNTERS.json" > "$log" 2>&1 || rc=$?
  secs="$(( $(date +%s) - t0 ))"
  out="$("$PY_BASE" "$HELPERS" counters "$SESSION" "$log" "$rc" "$cap" "$secs" \
         "$bin" "$where" "$cands" "$NCU_SEARCH" 2>&1)" || hrc=$?
  note="$(printf '%s' "${out##*$'\n'}" | tr '\t' ' ')"
  if (( hrc != 0 )) || [[ -z "$note" ]]; then
    note="ERROR: the counters helper exited $hrc (${note:-no output}); read $log"
  fi
  printf '%s\tINFO\t%s\t%s\t%s\t%s\t%s\n' counter-probe "$rc" "$secs" "$(dirty_count)" \
    "$log" "$note" >> "$LEDGER"
  printf '%-16s %-10s rc=%s %5ss  %s\n' counter-probe INFO "$rc" "$secs" "$note"
  return 0
}

#: THE DIRECTORY TO RESUME: the newest chain session for this card holding a
#: measuring ledger, or rc 1. By name is by time (the names carry a UTC
#: stamp). A dry run's directory holds CHAIN-dryrun.tsv only and latches
#: nothing: the driver's latest_session, lifted, which the chain lacked.
latest_chain_session() {
  local root="$1" prefix="$2" d found=""
  for d in "$root/$prefix"*/; do
    [[ -f "$d/CHAIN.tsv" ]] && found="${d%/}"
  done
  [[ -n "$found" ]] || return 1
  printf '%s\n' "$found"
}

#: WHICH DIRECTORY THIS PASS RUNS IN, the driver's session_choice with the
#: chain's two extra rules. $1 DRY, $2 --resume, $3 --new, $4 SESSION= as
#: given, $5 the root, $6 the per-card prefix. Prints HOW<TAB>WHAT; rc 1 on:
#:   REFUSED        contradictory flags; a dry run pointed at a real session
#:                  (it would overwrite its logs with plan pages); nothing to
#:                  resume (the directories found are listed)
#:   LATEST_EXISTS  a measuring run without --new found a chain session
#: and rc 0 on NAMED, RESUMED and NEW.
session_choice() {
  local dry="$1" resume="$2" new="$3" explicit="$4" root="$5" prefix="$6" latest="" d others=""
  if (( resume )) && (( new )); then
    printf 'REFUSED\t--resume and --new contradict each other. Give one.\n'; return 1
  fi
  if (( dry )) && { (( resume )) || [[ -n "$explicit" ]]; }; then
    printf 'REFUSED\t--dry-run plans into a fresh directory of its own. With --resume or SESSION= it would overwrite that session'"'"'s chain-logs, and the driver'"'"'s logs/thermal.log, with plan pages. Run the dry run alone.\n'
    return 1
  fi
  if [[ -n "$explicit" ]]; then
    if (( resume )) || (( new )); then
      printf 'REFUSED\tSESSION=%s names the directory outright, and --resume / --new choose one under %s. Give one or the other.\n' "$explicit" "$root"
      return 1
    fi
    printf 'NAMED\t%s\n' "$explicit"; return 0
  fi
  latest="$(latest_chain_session "$root" "$prefix")" || latest=""
  if (( resume )); then
    if [[ -z "$latest" ]]; then
      for d in "$root/$prefix"*/; do [[ -d "$d" ]] && others+=" ${d%/}"; done
      printf 'REFUSED\t--resume found no session holding CHAIN.tsv under %s/%s*; the directories there (dry runs, or never measured):%s\n' \
        "$root" "$prefix" "${others:- none}"
      return 1
    fi
    printf 'RESUMED\t%s\n' "$latest"; return 0
  fi
  if (( dry == 0 )) && (( new == 0 )) && [[ -n "$latest" ]]; then
    printf 'LATEST_EXISTS\t%s\n' "$latest"; return 1
  fi
  printf 'NEW\t%s/%s%s\n' "$root" "$prefix" "$(date -u +%Y%m%dT%H%M%SZ)"
}

#: THE CARD THIS SESSION WAS MEASURED ON. $1 the session, $2 this card's
#: identity. Writes DEVICE on first use; returns 2, saying why, when the
#: session was measured on another card, when its ledger has rows but no
#: DEVICE (the card those rows came from is unknown), or when this card's
#: UUID cannot be read. Both arms guard their own directories on the UUID
#: too (R3's DEVICE file, R1's device_guard since 858bf35), so this check
#: duplicates them on purpose: it refuses before any step runs, on the
#: session as a whole, where theirs refuse one run directory at a time.
device_check() {
  local file="$1/DEVICE" ident="$2" recorded rows
  if [[ -z "$ident" ]]; then
    echo "REFUSED: this card's UUID could not be read (torch and nvidia-smi gave none), so"
    echo "  a resume here could not be shown to be on the card that began it."
    return 2
  fi
  if [[ -f "$file" ]]; then
    recorded="$(tr -d '[:space:]' < "$file")"
    [[ "$recorded" == "$ident" ]] && return 0
    echo "REFUSED: $1 was measured on the card $recorded, and this card is $ident."
    echo "  One session is one card: R3's and R1's own UUID guards would refuse each run"
    echo "  directory that card began, one step at a time. A fresh session on this card,"
    echo "  on purpose:"
    echo "      bash scripts/alpha_g_chain.sh --new"
    return 2
  fi
  rows="$(awk 'NR > 1' "$1/CHAIN.tsv" 2>/dev/null | grep -c . || true)"
  if (( ${rows:-0} > 0 )); then
    echo "REFUSED: $1 holds $rows ledger row(s) and no DEVICE file, so the card they were"
    echo "  measured on is unknown. A fresh session: bash scripts/alpha_g_chain.sh --new"
    return 2
  fi
  printf '%s\n' "$ident" > "$file"
}

#: THE DUTY THIS SESSION'S RATIO RUNS ARE AT. $1 the session, $2 R3_DUTY.
#: Writes R3_DUTY on the first measuring pass, beside DEVICE; returns 2,
#: saying why, when the session recorded another duty, or holds ratio rows and
#: no record. Duty is one of R3's design keys: a --resume at another one left
#: every latched seed 0 at the old duty, read the old seed 0's V7 for the skip,
#: and handed every owed later seed earlier seeds its load_replicates refuses,
#: so it measured nothing new and read as if it had.
duty_check() {
  local file="$1/R3_DUTY" duty="$2" recorded rows
  if [[ -f "$file" ]]; then
    recorded="$(tr -d '[:space:]' < "$file")"
    awk -v a="$recorded" -v b="$duty" 'BEGIN {exit !(a + 0 == b + 0 && a != "")}' && return 0
    echo "REFUSED: $1's ratio runs are at --duty ${recorded:-(an empty record)} ($file), and"
    echo "  this pass asks for R3_DUTY=$duty. Duty is a design key of R3: seed 0 stays latched"
    echo "  at the recorded duty and every owed later seed would be refused against it."
    echo "  Resume this session at its own duty (without R3_DUTY=), or open a session at"
    echo "  the new one, on purpose:"
    echo "      R3_DUTY=$duty bash scripts/alpha_g_chain.sh --new"
    echo "  A lower duty for one G after a V7 FAIL is the follow-up the chain printed,"
    echo "  run by hand after it, not a resume."
    return 2
  fi
  rows="$(awk -F'\t' 'NR > 1 && $1 ~ /^r3-g/' "$1/CHAIN.tsv" 2>/dev/null | grep -c . || true)"
  if (( ${rows:-0} > 0 )); then
    echo "REFUSED: $1 holds $rows ratio row(s) and no R3_DUTY file, so the duty they ran at"
    echo "  is unknown here. PAIRS-fixed.tsv's r3_duty row reads it off the reports; write"
    echo "  that value into $file and --resume, or open a fresh session:"
    echo "      bash scripts/alpha_g_chain.sh --new"
    return 2
  fi
  printf '%s\n' "$duty" > "$file"
}

#: Is a recorded mkdir-lock owner ("pid host") gone: its host differs from
#: this one (the pod it ran on is not this pod), or its pid is dead here. $1
#: the owner, $2 this host. Returns 0 when stale. The flock branch never asks:
#: a dead recorded pid says nothing about the arm that pid left running.
lock_is_stale() {
  local pid="${1%% *}" host="${1#* }"
  [[ -n "$1" && "$1" == *" "* ]] || return 0
  [[ "$host" != "$2" ]] && return 0
  kill -0 "$pid" 2>/dev/null && return 1
  return 0
}

#: THE LOCK, held for the whole measuring run. flock where the box has it,
#: else an atomic mkdir; either way the holder is recorded as "pid host". $1
#: the session, $2 1 when this pass continues an existing session. Returns 2,
#: saying why, when refused; sets LOCK_DIR to what release_lock removes
#: (nothing for flock, which the kernel releases when this shell and every
#: child holding fd 9 are gone).
#:   flock  A held lock is NEVER taken over, whatever pid it records: the
#:          kernel's answer is that a live process holds the file, and when
#:          the recorded chain is dead that process is the arm it left running
#:          (`pkill -f alpha_g_chain.sh` kills the shell, not its python),
#:          which inherited fd 9 and is still timing the card. A resume that
#:          took it over would run the same step, same run id, on the same
#:          card beside it.
#:   mkdir  The fallback where there is no flock (a laptop). It cannot see an
#:          orphaned arm, so a resuming pass ($2 1) takes over a lock whose
#:          pid is dead or whose host differs.
LOCK_TOOL="${LOCK_TOOL:-}"
[[ -n "$LOCK_TOOL" ]] || { command -v flock >/dev/null 2>&1 && LOCK_TOOL=flock || LOCK_TOOL=mkdir; }
LOCK_DIR=""
LOCK_OWNER=""
chain_lock() {
  local dir="$1" resuming="$2" host owner
  host="$(hostname 2>/dev/null || uname -n 2>/dev/null || echo unknown-host)"
  LOCK_OWNER="$$ $host"
  if [[ "$LOCK_TOOL" == flock ]]; then
    local file="$dir/chain.lock" frc=0
    exec 9>>"$file" || { echo "REFUSED: cannot open $file"; return 2; }
    flock -n 9 || frc=$?
    # util-linux flock exits 1 on a CONFLICT (its -E default) and an EX_* code
    # (64 and up) when it could not lock at all, which is what a network
    # filesystem without flock support answers. Only 1 means a holder; any
    # other failure falls back to the mkdir lock and says so, rather than
    # refusing every measuring run on a volume that cannot flock.
    if (( frc != 0 && frc != 1 )); then
      exec 9>&-
      echo "  flock could not lock $file (exit $frc: this filesystem does not flock);"
      echo "  falling back to the mkdir lock, which cannot see an orphaned arm"
      LOCK_TOOL=mkdir
    elif (( frc == 1 )); then
      owner="$(cat "$file" 2>/dev/null)"
      exec 9>&-
      echo "REFUSED: another chain holds $file (it recorded ${owner:-no holder}); two chains"
      echo "  on one session would time the card at once and resume the same run ids."
      echo "  The kernel says a live process holds it, whatever pid it records: a chain, or"
      echo "  an arm a killed chain left running, still timing the card. A held flock is"
      echo "  never taken over. Find the holder, stop it, then --resume:"
      echo "      fuser -v $file     (or: lsof $file)"
      echo "      ps -eo pid,etime,args | grep -E '[a]lpha_g_chain[.]sh|[p]rivate_weight_reference|[c]lock_elasticity|[h]200_gaps_session'"
      echo "  fuser and lsof may be missing from the image; the ps line always runs. Only when"
      echo "  the ps line names nothing here is the holder on another pod sharing this volume."
      echo "  Do not open a --new session meanwhile: its arms would time this card beside"
      echo "  whatever still holds the lock."
      return 2
    else
      printf '%s\n' "$LOCK_OWNER" > "$file"
      return 0
    fi
  fi
  local ldir="$dir/chain.lock.d"
  if ! mkdir "$ldir" 2>/dev/null; then
    owner="$(cat "$ldir/owner" 2>/dev/null)"
    if (( resuming )) && lock_is_stale "$owner" "$host"; then
      rm -rf "$ldir"
      mkdir "$ldir" 2>/dev/null || { echo "REFUSED: lost the race for $ldir"; return 2; }
      echo "  took over a stale lock (held by ${owner:-nobody recorded})"
    else
      echo "REFUSED: another chain holds $ldir (${owner:-no holder recorded}); two chains"
      echo "  on one session would time the card at once and resume the same run ids."
      echo "  If no chain is running, a --resume takes a lock whose pid is dead over"
      echo "  (this box has no flock, so an arm a killed chain left running is not seen)."
      return 2
    fi
  fi
  printf '%s\n' "$LOCK_OWNER" > "$ldir/owner"
  LOCK_DIR="$ldir"
  return 0
}

#: The EXIT trap's half of the mkdir lock: removed only while it still names
#: this shell. A pass that took the lock over from a holder it judged stale
#: must not lose it when that holder, alive after all on another host, exits.
release_lock() {
  [[ -n "$LOCK_DIR" ]] || return 0
  [[ "$(cat "$LOCK_DIR/owner" 2>/dev/null)" == "$LOCK_OWNER" ]] && rm -rf "$LOCK_DIR"
  return 0
}
# the command lines, in one place each (TAG is the session's name)
r1_cmd() {   # $1 G, $2 dry
  local g="$1" dry="$2"
  if (( dry )); then
    echo "$PY_BASE" "$REPO/scripts/clock_elasticity.py" --dry-run
  else
    echo "$PY_VLLM" "$REPO/scripts/clock_elasticity.py"
  fi
  # shellcheck disable=SC2086
  echo --model mixtral-8x7b --dtype bf16 --group-m "$g" --treads "$R1_TREADS" --duty $R1_DUTY \
       --repeats "$R1_REPEATS" --burst-ms 40 --target-ms 200 --trials 3 --warm-ms 200 \
       --settle-seconds 10 --session-tag "$TAG"
}
#: The ratio arm at a named duty: the chain's own R3 flags, so a follow-up at
#: R3_FOLLOWUP_DUTY differs from the chain's runs in the duty and nothing else.
r3_cmd_at() {   # $1 duty, $2 G, $3 seed, $4 dry, $5.. replicate reports
  local duty="$1" g="$2" seed="$3" dry="$4"; shift 4
  if (( dry )); then
    echo "$PY_BASE" "$REPO/scripts/private_weight_reference.py" --dry-run \
         --capability "${CAPABILITY:-9.0}" --device-memory-gb 140
  else
    echo "$PY_VLLM" "$REPO/scripts/private_weight_reference.py"
  fi
  echo --model mixtral-8x7b --block-m 32 --treads "$R3_TREADS" --repeats "$R3_REPEATS" \
       --group-m "$g" --duty "$duty" --seed "$seed" --session-tag "$TAG"
  if (( $# )); then echo --replicate-of "$@"; fi
}
r3_cmd() { r3_cmd_at "$R3_DUTY" "$@"; }   # $1 G, $2 seed, $3 dry, $4.. replicate reports
probe_check_cmd() {
  echo "$PY_VLLM" "$REPO/scripts/private_weight_reference.py" --probe-check \
       --model mixtral-8x7b --block-m 32
}

#: The price of a follow-up for G $1: its seeds at R3_FOLLOWUP_DUTY, off R3's
#: own plan at that duty, run into chain-logs. Prints "<seconds a run><TAB>
#: <runs><TAB><the plan's seconds>", the run the plan plus R3_RUN_OVERHEAD_S.
followup_price() {
  local g="$1" est n=0 s
  for s in $SEEDS; do n=$(( n + 1 )); done
  # shellcheck disable=SC2046
  est="$(plan_price "$LOGS/followup-g$g.price.log" $(r3_cmd_at "$R3_FOLLOWUP_DUTY" "$g" "$FIRST_SEED" 1))"
  printf '%s\t%s\t%s\n' "$(( ${est:-0} + R3_RUN_OVERHEAD_S ))" "$n" "${est:-unpriced}"
}

#: THE FOLLOW-UP A V7 FAIL AT SEED 0 NAMES, for G $1, per the owner's reading
#: of 2026-09-22: a V7 FAIL at the chain's duty means that duty is not yet flat
#: for that arm on this card, and R3's page names a lower one. The chain does
#: not act on it: it skips the G's later seeds and prints, here and in
#: $SESSION/followup-g<G>.txt, that G's seeds at R3_FOLLOWUP_DUTY with the
#: chain's own R3 flags and session tag, to run by hand AFTER the chain (never
#: beside it: two runs would time the card at once). Later seeds are scored
#: with the follow-up's first seed through --replicate-of. It is a new design
#: key and its own runs, read with --read on the laptop, outside PAIRS.tsv.
#: Sets FOLLOWUP_NOTE, the ledger's one-line form.
FOLLOWUP_NOTE=""
followup() {
  local g="$1" file per n plan first s0log slog s
  if ! awk -v f="$R3_FOLLOWUP_DUTY" -v d="$R3_DUTY" 'BEGIN {exit !(f + 0 < d + 0)}'; then
    FOLLOWUP_NOTE="no follow-up duty below $R3_DUTY is registered in this chain (R3_FOLLOWUP_DUTY is $R3_FOLLOWUP_DUTY); the page's V7 lines are the record"
    echo "    $FOLLOWUP_NOTE"
    return 0
  fi
  # read from a string, not `< <(...)`: a process substitution on a command
  # with IFS=<tab> in front of it runs with that IFS, and no word would split
  local priced
  priced="$(followup_price "$g")"
  IFS=$'\t' read -r per n plan <<< "$priced"
  file="$SESSION/followup-g$g.txt"
  s0log="$SESSION/followup-logs/r3-g$g-s$FIRST_SEED-duty$R3_FOLLOWUP_DUTY.log"
  # shellcheck disable=SC2046
  first="$(echo $(r3_cmd_at "$R3_FOLLOWUP_DUTY" "$g" "$FIRST_SEED" 0)) > $s0log 2>&1"
  {
    echo "# G=$g: seed $FIRST_SEED's page (r3-g$g-s$FIRST_SEED) read V7 FAIL at --duty $R3_DUTY. A V7 FAIL at"
    echo "# $R3_DUTY means that duty is not yet flat for that arm on this card; the page names a"
    echo "# lower duty. These lines run G=$g's seeds at --duty $R3_FOLLOWUP_DUTY: a new design key and"
    echo "# its own runs, outside the chain's ledger and PAIRS.tsv."
    echo "# RUN THEM AFTER THE CHAIN HAS FINISHED, never beside it: two runs would time the"
    echo "# card at once. ~$(( (n * per + 59) / 60 )) min: $n runs at $per s each, R3's own plan at --duty"
    echo "# $R3_FOLLOWUP_DUTY ($plan s) plus $R3_RUN_OVERHEAD_S s of compiles and weight build."
    echo "cd $REPO"
    echo "export MOE_RESULTS_DIR=$RESULTS"
    echo "mkdir -p $SESSION/followup-logs"
    echo "$first"
    echo "S0=\"\$MOE_RESULTS_DIR/private_weight_reference/\$($PY_BASE $HELPERS run-id $s0log private_weight_reference)/report.json\""
    for s in $SEEDS; do
      [[ "$s" == "$FIRST_SEED" ]] && continue
      slog="$SESSION/followup-logs/r3-g$g-s$s-duty$R3_FOLLOWUP_DUTY.log"
      # shellcheck disable=SC2046
      echo "$(echo $(r3_cmd_at "$R3_FOLLOWUP_DUTY" "$g" "$s" 0 '"$S0"')) > $slog 2>&1"
    done
    echo "# The chain's exfil line carries them (under $SESSION and $RESULTS). Then, on the"
    echo "# laptop, outside PAIRS.tsv, the seeds read together:"
    echo "#   .venv/bin/python scripts/private_weight_reference.py --read <a later seed's run>/report.json --replicate-of <the other runs>/report.json"
  } > "$file"
  sed 's/^/    /' "$file"
  FOLLOWUP_NOTE="the follow-up, by hand AFTER the chain and never beside it: G=$g's seeds at --duty $R3_FOLLOWUP_DUTY, ~$(( (n * per + 59) / 60 )) min off R3's own plan at $R3_FOLLOWUP_DUTY, a new design key read with --read on the laptop, outside PAIRS.tsv; every line is in $file, the first: $first"
}
# <<< LIFTABLE

usage() { sed -n '2,/^set -uo pipefail$/p' "$0" | sed '$d'; }

DRY=0; RESUME=0; NEW=0; PAST_GPU_TESTS=0; PAST_V8=0
while (( $# )); do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --resume)  RESUME=1; shift ;;
    --new)     NEW=1; shift ;;
    --past-gpu-tests) PAST_GPU_TESTS=1; shift ;;
    --past-v8) PAST_V8=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "REFUSED: unknown argument $1"; usage; exit 2 ;;
  esac
done

[[ -x "$PY_BASE" ]] || { echo "REFUSED: no usable base interpreter (set PY_BASE=)"; exit 2; }
[[ -f "$HELPERS" ]] || { echo "REFUSED: $HELPERS is missing"; exit 2; }
case "$END_SUITE" in
  run|skip) ;;
  *) echo "REFUSED: END_SUITE=$END_SUITE: it takes run (buy the end suite after the arms) or"
     echo "  skip (the default: not requested, a SKIPPED row). Nothing was run and no session"
     echo "  directory was opened."
     exit 2 ;;
esac

smi_line() {
  local out
  if command -v nvidia-smi >/dev/null 2>&1; then
    out="$(nvidia-smi --query-gpu=name,driver_version,power.limit --format=csv,noheader 2>&1)" \
      || out="nvidia-smi failed: $out"
    printf '%s\n' "$out" | sed 's/^/  nvidia-smi  /'
  else
    echo "  nvidia-smi  not on PATH (name, driver version and power limit unread)"
  fi
}

# THE CARD, as the driver resolves it: through torch on PY_BASE, as a slug,
# with the probe's own reason when it cannot, and its stderr NOT discarded.
CARD=""; CARD_REASON=""
IFS=$'\t' read -r CARD CARD_REASON < <("$PY_BASE" "$HELPERS" card)
CARD="${CARD:-nocard}"
[[ "$CARD" == nocard ]] && CARD_REASON="${CARD_REASON:-the card probe printed nothing}"
if [[ "$CARD" == nocard ]] && ! (( DRY )); then
  echo "REFUSED: no CUDA device this chain can name, and a measuring run measures."
  echo "  Nothing was run and no session directory was opened."
  echo "  the probe ($PY_BASE, through moe.bench.provenance) said: $CARD_REASON"
  smi_line
  echo "  A torch wheel built for a newer CUDA than the host's driver reads as no device:"
  echo "  compare the driver version above with the wheel's. Off GPU and free:"
  echo "      bash scripts/alpha_g_chain.sh --dry-run"
  exit 2
fi

if [[ -d /workspace ]]; then
  SESSION_ROOT="${SESSION_ROOT:-/workspace/session}"
  RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results}"
  WORKSPACE="${WORKSPACE:-/workspace}"
else
  SESSION_ROOT="${SESSION_ROOT:-$REPO/results/h200_gaps}"
  RESULTS_ROOT="${RESULTS_ROOT:-$REPO/results}"
  WORKSPACE="${WORKSPACE:-$RESULTS_ROOT}"
fi
IFS=$'\t' read -r SESSION_HOW SESSION_WHAT \
  <<< "$(session_choice "$DRY" "$RESUME" "$NEW" "${SESSION:-}" "$SESSION_ROOT" "alpha_g-$CARD-")"
case "$SESSION_HOW" in
  NAMED|RESUMED|NEW) SESSION="$SESSION_WHAT" ;;
  LATEST_EXISTS)
    echo "REFUSED: a chain session for $CARD already exists, with a measuring ledger:"
    echo "  $SESSION_WHAT"
    echo "  A fresh session opens an EMPTY ledger under a new tag, and every R1 and R3"
    echo "  run id carries the tag, so every arm that session finished would be measured"
    echo "  again. Nothing was run. Choose, in words:"
    echo "      bash scripts/alpha_g_chain.sh --resume     # continue it"
    echo "      SESSION=$SESSION_WHAT bash scripts/alpha_g_chain.sh"
    echo "      bash scripts/alpha_g_chain.sh --new        # a fresh session, on purpose"
    echo "  Before --new after a killed chain: an arm it left running still times the"
    echo "  card. Look first: ps -eo pid,etime,args | grep -E '[p]rivate_weight_reference|[c]lock_elasticity'"
    exit 2 ;;
  *)
    echo "REFUSED: ${SESSION_WHAT:-session_choice printed nothing this chain can read}"
    exit 2 ;;
esac
CONTINUING=0
[[ "$SESSION_HOW" == NEW ]] || CONTINUING=1
# The driver's rule, applied here BEFORE the driver is reached and before any
# directory is made: an operator-supplied MOE_RESULTS_DIR must carry the card
# in its name. Two pods share a network volume, and one card's run resumes the
# other's directories when the card is not in the path.
if [[ -n "${MOE_RESULTS_DIR:-}" ]]; then
  RESULTS="$MOE_RESULTS_DIR"
  case "$RESULTS" in
    *"$CARD"*) ;;
    *) echo "REFUSED: MOE_RESULTS_DIR=$RESULTS does not contain the card '$CARD'."
       echo "  Two pods share a network volume and one card's run resumes the"
       echo "  other's directories when the card is not in the path. Use"
       echo "  MOE_RESULTS_DIR=$RESULTS/gaps-$CARD or unset it and take the default."
       exit 2 ;;
  esac
else
  RESULTS="$RESULTS_ROOT/gaps-$CARD"
fi
export MOE_RESULTS_DIR="$RESULTS"
TAG="$(basename "$SESSION")"
LOGS="$SESSION/chain-logs"
if (( DRY )); then LEDGER="$SESSION/CHAIN-dryrun.tsv"; else LEDGER="$SESSION/CHAIN.tsv"; fi
mkdir -p "$LOGS" || { echo "REFUSED: cannot create $LOGS"; exit 2; }

if ! (( DRY )); then
  chain_lock "$SESSION" "$CONTINUING" || exit 2
  trap release_lock EXIT
  DEVICE_ID="$("$PY_BASE" "$HELPERS" device)" || DEVICE_ID=""
  device_check "$SESSION" "$DEVICE_ID" || exit 2
  duty_check "$SESSION" "$R3_DUTY" || exit 2
fi
[[ -f "$LEDGER" ]] || printf 'step\tstate\trc\tseconds\tdirty\tlog\tnote\n' > "$LEDGER"
#: Where the documented detached launch appends this chain's console.
CONSOLE_OUT="$WORKSPACE/alpha_g_chain.out"

echo "alpha(G) chain  $TAG   ($SESSION_HOW)"
echo "  card        $CARD${CARD_REASON:+  -- $CARD_REASON}"
smi_line
(( DRY )) || echo "  device      ${DEVICE_ID:-}   (recorded in $SESSION/DEVICE)"
echo "  session     $SESSION"
echo "  results     $RESULTS   (exported as MOE_RESULTS_DIR to every arm)"
echo "  ledger      $LEDGER"
echo "  ladder      G in {$G_LADDER}, seeds {$SEEDS}, ratio arms at duty $R3_DUTY, elasticity states $R1_DUTY"
(( DRY )) || echo "  r3 duty     $R3_DUTY   (recorded in $SESSION/R3_DUTY; a resume at another is refused)"
echo "  interpreters base $PY_BASE / vllm $PY_VLLM"
(( DRY )) && echo "  DRY RUN: every arm's own --dry-run is run and priced; nothing is measured"
if ! (( DRY )) && [[ -t 1 ]]; then
  echo "  NOTE: attached to a terminal. A dropped ssh session kills this chain and the arm"
  echo "  in flight; the launch line in this file's header detaches it."
fi

#: Every STOP after the session exists leaves the table as the disk has it.
stop_chain() { rebuild_pairs; exit 3; }

TOTAL_S=0; LAST_EST=""
GPU_TESTS_S=0; GPU_TESTS_N=""; SUITE_S=0; SUITE_N=""
#: The dry run's TIMELINE: every step's start on the priced clock, so the seed
#: spacing it prints is these prices' and not a sentence's.
CLOCK_S=0
mark() { printf -v "AT_${1//-/_}" '%s' "$CLOCK_S"; }
price() {   # $1 log: add the arm's own dry-run estimate to the total; say what it is and its cap
  local basis cap how
  LAST_EST="$("$PY_BASE" "$HELPERS" estimate "$1" 2>/dev/null)" || LAST_EST=""
  if [[ -n "$LAST_EST" ]]; then
    TOTAL_S=$(( TOTAL_S + LAST_EST ))
    basis="$("$PY_BASE" "$HELPERS" estimate-basis "$1" 2>/dev/null)" || basis=""
    echo "    priced ${LAST_EST} s off its own plan${basis:+: $basis}"
  else
    echo "    (no estimate on its plan page)"
  fi
  IFS=$'\t' read -r cap how < <(cap_for "$LAST_EST")
  echo "    capped on the pod at $cap s ($how)"
}
#: A measuring arm's cap, off its own plan run again just before it. $1 the
#: step, $2.. the dry command. Sets STEP_CAP and says so.
STEP_CAP=0
step_cap() {
  local step="$1" est how; shift
  est="$(plan_price "$LOGS/$step.price.log" "$@")"
  IFS=$'\t' read -r STEP_CAP how < <(cap_for "$est")
  echo "  $step: capped at $STEP_CAP s ($how)"
}

#: One ratio run: pairing with the earlier seeds of its G that FORMED a ratio
#: (the rule load_replicates applies), the step, and its console line.
r3_step() {   # $1 G, $2 seed
  local g="$1" seed="$2" step s2 elog rep line
  local -a earlier=() paired=()
  step="r3-g$g-s$seed"
  if latched "$step" "$LEDGER" && ! (( DRY )); then echo "  $step latched, skipped"; return 0; fi
  for s2 in $SEEDS; do
    [[ "$s2" == "$seed" ]] && break
    elog="$LOGS/r3-g$g-s$s2.log"
    [[ -f "$elog" ]] || continue
    rep="$(report_of "$elog")"
    [[ -n "$rep" ]] && earlier+=("$rep")
  done
  if (( ${#earlier[@]} )); then
    while IFS= read -r line; do [[ -n "$line" ]] && paired+=("$line"); done \
      < <("$PY_BASE" "$HELPERS" pairs ${earlier[@]+"${earlier[@]}"} 2>/dev/null)
  fi
  if (( DRY )); then
    mark "$step"
    # shellcheck disable=SC2046
    run_step "$step" "$LOGS/$step.log" $(r3_cmd "$g" "$seed" 1 ${paired[@]+"${paired[@]}"}) || true
    price "$LOGS/$step.log"
    CLOCK_S=$(( CLOCK_S + ${LAST_EST:-0} + R3_RUN_OVERHEAD_S ))
    (( ${#paired[@]} )) && echo "    scored WITH ${#paired[@]} earlier seed(s)"
    return 0
  fi
  # shellcheck disable=SC2046
  step_cap "$step" $(r3_cmd "$g" "$seed" 1)
  # shellcheck disable=SC2046
  arm_step "$step" "$LOGS/$step.log" "$STEP_CAP" $(r3_cmd "$g" "$seed" 0 ${paired[@]+"${paired[@]}"}) || true
  rep="$(report_of "$LOGS/$step.log")"
  if [[ -n "$rep" ]]; then
    local rg rseed ratio lo hi word scope duty rid rest e elo ehi band eexit ra
    IFS=$'\t' read -r rg rseed ratio lo hi word scope duty rid rest < <("$PY_BASE" "$HELPERS" reading "$rep")
    IFS=$'\t' read -r e elo ehi band eexit < <("$PY_BASE" "$HELPERS" eta-for "$SESSION" "$RESULTS" "$g")
    ra="$("$PY_BASE" "$HELPERS" reads-as "$band" "$eexit" 2>/dev/null)" || ra="unresolved"
    echo "    G=$rg seed $rseed  ratio $ratio [$lo, $hi]  $word (C1 $scope)  duty $duty  eta $e [$elo, $ehi] $band  (R1 page: $eexit)  reads as: ${ra:-unresolved}"
  else
    echo "    no report.json for $step (see $LOGS/$step.log)"
  fi
}

# --------------------------------------------------------------------------
# 1. preflight: the scorers, proven on this interpreter
# --------------------------------------------------------------------------
echo; echo "== preflight"
# A self-test that is not DONE runs again on every pass, in seconds on the
# CPU: a latched INVALID used to STOP every --resume the same way, even after
# the scorer was fixed and checked out on the pod.
if [[ "$(newest_state preflight-r1 "$LEDGER")" != DONE ]] || (( DRY )); then
  run_step preflight-r1 "$LOGS/preflight-r1.log" "$PY_BASE" "$REPO/scripts/clock_elasticity.py" --self-test --draws 50 || true
fi
if [[ "$(newest_state preflight-r3 "$LEDGER")" != DONE ]] || (( DRY )); then
  run_step preflight-r3 "$LOGS/preflight-r3.log" "$PY_BASE" "$REPO/scripts/private_weight_reference.py" --self-test refit || true
fi
(( DRY )) || preflight_gate || stop_chain

# --------------------------------------------------------------------------
# 2. preconditions, through the driver, in this session directory
# --------------------------------------------------------------------------
echo; echo "== preconditions (the driver's thermal, calibrate; pin_probe-n64-g1 for the record)"
if ! latched preconditions "$LEDGER" || (( DRY )); then
  pre_flags=(--only thermal,calibrate,pin_probe-n64-g1)
  (( DRY )) && pre_flags+=(--dry-run)
  run_step_as driver preconditions "$LOGS/preconditions.log" \
    env SESSION="$SESSION" bash "$REPO/scripts/h200_gaps_session.sh" "${pre_flags[@]}" || true
fi
(( DRY )) || preconditions_gate "$SESSION/ARMS.tsv" || stop_chain
if (( DRY )); then
  # the price of the same three arms is the driver's own booking, not a guess
  PRE_S=0; PRE_BASIS=""
  for arm in thermal calibrate pin_probe-n64-g1; do
    m="$(driver_minutes "$arm")"
    PRE_S=$(( PRE_S + ${m:-0} * 60 ))
    PRE_BASIS+="${PRE_BASIS:+, }$arm ${m:-unbooked}"
  done
  PRE_BASIS="the driver's own arm_minutes: $PRE_BASIS min"
  echo "    priced ${PRE_S} s: $PRE_BASIS"
  CLOCK_S=$(( CLOCK_S + PRE_S ))
fi

# --------------------------------------------------------------------------
# 3. the counter probe: can this pod read a DRAM counter (informational)
# --------------------------------------------------------------------------
echo; echo "== counter-probe: can this pod read a DRAM counter (informational; gates nothing)"
COUNTER_PROBE_S="$(counter_probe_price)"
if (( DRY )); then
  IFS=$'\t' read -r CP_CAP CP_HOW < <(cap_for "$COUNTER_PROBE_S")
  skip_row counter-probe "a dry run does not run it: it launches one kernel under ncu on the card. Priced ~${COUNTER_PROBE_S:-0} s, the driver's own arm_minutes for counter_plan (the same probe); capped on the pod at $CP_CAP s ($CP_HOW); informational, it gates nothing and runs on every measuring pass"
  CLOCK_S=$(( CLOCK_S + ${COUNTER_PROBE_S:-0} ))
else
  # never gated, never latched: whatever it reads, the chain goes on
  counter_probe_step "$COUNTER_PROBE_S"
fi

# --------------------------------------------------------------------------
# 4. tests/test_gpu.py on this card, before any arm is booked on it
# --------------------------------------------------------------------------
echo; echo "== tests/test_gpu.py on this card, from the base venv (gated)"
if (( DRY )); then
  # a dry run COLLECTS and prices; off a card every one of them would skip
  run_step_as collect gpu-tests "$LOGS/gpu-tests.log" in_repo without_knobs "$PY_BASE" -m pytest \
    tests/test_gpu.py --collect-only -q -p no:cacheprovider || true
  read -r GPU_TESTS_N GPU_TESTS_S < <(price_tests "$LOGS/gpu-tests.log")
  echo "    priced ${GPU_TESTS_S} s: $GPU_TESTS_N tests at $SUITE_S_PER_TEST s each, $SUITE_RATE_WHO"
  CLOCK_S=$(( CLOCK_S + GPU_TESTS_S ))
elif ! latched gpu-tests "$LEDGER" && ! (( PAST_GPU_TESTS )) && ! overridden gpu-tests-override; then
  # --past-gpu-tests is a decision taken AFTER reading a red page: it does not
  # run the same file again, on this pass or a later one
  pytest_step gpu-tests "$GPU_TESTS_TIMEOUT_S" tests/test_gpu.py -q -rfE -p no:cacheprovider
fi
(( DRY )) || gpu_tests_gate "$PAST_GPU_TESTS" || stop_chain

# --------------------------------------------------------------------------
# 5. the alignment probe under the graph, on this card, before the pilot
# --------------------------------------------------------------------------
echo; echo "== the alignment probe under the graph, on this card, from the vllm venv (gated)"
IFS=$'\t' read -r PROBE_CAP PROBE_CAP_HOW < <(cap_for "$PROBE_CHECK_S")
if (( DRY )); then
  skip_row probe-check "a dry run does not run it: it times the card. Priced ~$PROBE_CHECK_S s, a documented allowance (it prints no plan); capped on the pod at $PROBE_CAP s ($PROBE_CAP_HOW)"
  CLOCK_S=$(( CLOCK_S + PROBE_CHECK_S ))
elif [[ "$(newest_state probe-check "$LEDGER")" != DONE ]] && ! (( PAST_V8 )) \
     && ! overridden probe-check-override; then
  # runs again until DONE, as a preflight does: its remedy is a code change,
  # and a --resume after it must prove the probe again
  echo "  probe-check: capped at $PROBE_CAP s ($PROBE_CAP_HOW, over its documented allowance)"
  # shellcheck disable=SC2046
  arm_step probe-check "$LOGS/probe-check.log" "$PROBE_CAP" $(probe_check_cmd) || true
fi
(( DRY )) || probe_check_gate "$PAST_V8" || stop_chain

# --------------------------------------------------------------------------
# 6. the ratio at seed 0 at every G; the pilot's V8 read first
# --------------------------------------------------------------------------
echo; echo "== the ratio, shared over private, off the cap, seed $FIRST_SEED at every G (the pilot: $PILOT)"
for g in $G_LADDER; do
  r3_step "$g" "$FIRST_SEED"
  if [[ "$g" == "$FIRST_G" ]] && ! (( DRY )); then
    v8_gate "$PAST_V8" || stop_chain
  fi
done

# --------------------------------------------------------------------------
# 7. the elasticity at every G of the ladder
# --------------------------------------------------------------------------
echo; echo "== clock elasticity per geometry (the regime word)"
for g in $G_LADDER; do
  step="r1-g$g"
  if latched "$step" "$LEDGER" && ! (( DRY )); then echo "  $step latched, skipped"; continue; fi
  if (( DRY )); then
    mark "$step"
    # shellcheck disable=SC2046
    run_step "$step" "$LOGS/$step.log" $(r1_cmd "$g" 1) || true
    price "$LOGS/$step.log"
    CLOCK_S=$(( CLOCK_S + ${LAST_EST:-0} ))
  else
    # shellcheck disable=SC2046
    step_cap "$step" $(r1_cmd "$g" 1)
    # shellcheck disable=SC2046
    arm_step "$step" "$LOGS/$step.log" "$STEP_CAP" $(r1_cmd "$g" 0) || true
  fi
done

# --------------------------------------------------------------------------
# 8. the later seeds, seed-major, each scored with the earlier ones of its G
# --------------------------------------------------------------------------
echo; echo "== the ratio, shared over private, the later seeds, scored with the earlier ones"
for seed in $SEEDS; do
  [[ "$seed" == "$FIRST_SEED" ]] && continue
  for g in $G_LADDER; do
    step="r3-g$g-s$seed"
    if latched "$step" "$LEDGER" && ! (( DRY )); then echo "  $step latched, skipped"; continue; fi
    v7="$(seed0_v7 "$g")"
    if [[ -n "$v7" && "$v7" != PASS ]]; then
      # WORDED BY THE VERDICT: a FAIL is a clock split that was measured, and
      # an UNKNOWN (or no V7 at all) is a page that compared no clocks, which
      # is what every page whose sweep a V8 FAIL skipped reads
      if [[ "$v7" == FAIL ]]; then
        fu_var="FU_G$g"
        if [[ -z "${!fu_var:-}" ]]; then
          echo "  G=$g: seed $FIRST_SEED read V7 FAIL at --duty $R3_DUTY. Its follow-up, by hand AFTER the chain:"
          followup "$g"
          printf -v "$fu_var" '%s' "$FOLLOWUP_NOTE"
        fi
        note="seed $FIRST_SEED's page (r3-g$g-s$FIRST_SEED) read V7 FAIL: the two ratio arms ran at different clocks at duty $R3_DUTY, the power state, so a later seed at this duty would buy the same split. A V7 FAIL at $R3_DUTY means that duty is not yet flat for that arm on this card; the page names a lower duty; the chain skips the G's later seeds and prints the follow-up command. ${!fu_var}"
      else
        note="seed $FIRST_SEED's page (r3-g$g-s$FIRST_SEED) read V7 $v7: seed $FIRST_SEED's V7 could not be scored (nothing timed, e.g. the sweep was skipped on V8, or a clock was unread), so a later seed would read the same; see $LOGS/r3-g$g-s$FIRST_SEED.log"
      fi
      skip_row "$step" "$note"
      continue
    fi
    r3_step "$g" "$seed"
  done
done

# --------------------------------------------------------------------------
# 9. the whole suite, after every arm, only on END_SUITE=run: a record of
#    this box, gating nothing
# --------------------------------------------------------------------------
echo; echo "== the whole suite, uncapped, from the base venv (a record; gates nothing; only on END_SUITE=run)"
if (( DRY )) && [[ "$END_SUITE" == skip ]]; then
  skip_row suite "END_SUITE=skip (the default): the end suite was not requested; a dry run neither collects nor prices it, and END_SUITE=run does both"
elif (( DRY )); then
  # a dry run COLLECTS and prices; running it here would take the laptop 20
  # minutes and run this chain's own dry-run tests inside itself
  run_step_as collect suite "$LOGS/suite.log" in_repo without_knobs "$PY_BASE" -m pytest tests/ \
    --collect-only -q -p no:cacheprovider || true
  read -r SUITE_N SUITE_S < <(price_tests "$LOGS/suite.log")
  echo "    priced ${SUITE_S} s: $SUITE_N tests at $SUITE_S_PER_TEST s each, $SUITE_RATE_WHO"
elif SUITE_RECORD="$(end_suite_recorded)"; then
  # asked BEFORE END_SUITE=skip: a SKIPPED row over a DONE one made the next
  # plain --resume buy the whole suite again
  echo "  suite recorded, not bought again: $SUITE_RECORD"
elif [[ "$END_SUITE" == skip ]]; then
  skip_row suite "END_SUITE=skip (the default): the end suite was not requested this pass; END_SUITE=run buys it after the arms"
else
  pytest_step suite "$SUITE_TIMEOUT_S" tests/ -q -rfE --durations=25 -p no:cacheprovider
fi

# --------------------------------------------------------------------------
# 10. the table, the price, and what to copy off
# --------------------------------------------------------------------------
echo
if (( DRY )); then
  n_r1=0; for g in $G_LADDER; do n_r1=$(( n_r1 + 1 )); done
  n_s=0; for s in $SEEDS; do n_s=$(( n_s + 1 )); done
  n_r3=0; for s in $SEEDS; do for g in $G_LADDER; do n_r3=$(( n_r3 + 1 )); done; done
  compile_s=$(( n_r3 * R3_RUN_OVERHEAD_S ))
  wall=$(( TOTAL_S + GPU_TESTS_S + SUITE_S + PRE_S + ${COUNTER_PROBE_S:-0} + PROBE_CHECK_S + compile_s + EXFIL_S ))
  echo "PRICE, off the arms' own plans: $n_r1 elasticity runs + $n_r3 ratio runs = $TOTAL_S s of arms"
  echo "  (a ratio run is its plan's wall line at duty $R3_DUTY plus the alignment probe's"
  echo "  seconds at full duty; an elasticity run is its plan's wall figure at duty $R1_DUTY),"
  if [[ "$END_SUITE" == run ]]; then
    echo "  tests/test_gpu.py ~$GPU_TESTS_S s (${GPU_TESTS_N:-?} tests) before them and the end suite ~$SUITE_S s"
    echo "  (${SUITE_N:-0} tests) after them (END_SUITE=run), at $SUITE_S_PER_TEST s a test, $SUITE_RATE_WHO,"
  else
    echo "  tests/test_gpu.py ~$GPU_TESTS_S s (${GPU_TESTS_N:-?} tests) before them at $SUITE_S_PER_TEST s a test,"
    echo "  $SUITE_RATE_WHO, and no end suite (END_SUITE=skip, the default; END_SUITE=run prices it),"
  fi
  echo "  plus the preconditions ~$PRE_S s ($PRE_BASIS),"
  echo "  the counter probe ~${COUNTER_PROBE_S:-0} s (the driver's own arm_minutes for counter_plan, the same probe),"
  echo "  the probe check ~$PROBE_CHECK_S s (an allowance: it prints no plan), per-run compiles and"
  echo "  weight copies ~$compile_s s ($R3_RUN_OVERHEAD_S s a ratio run, an allowance) and exfil ~$EXFIL_S s"
  echo "  (an allowance) = $wall s"
  echo "  = $(( (wall + 59) / 60 )) min; at \$$RATE_USD_H/h about \$$(awk -v w="$wall" -v r="$RATE_USD_H" 'BEGIN {printf "%.2f", w / 3600 * r}'). Book $(( (wall + 3599) / 3600 + 1 )) h."
  # the seed spacing, start to start, off the timeline these prices drew
  spacing=""; prev=""
  for s in $SEEDS; do
    if [[ -n "$prev" ]]; then
      lo=""; hi=""
      for g in $G_LADDER; do
        a="AT_r3_g${g}_s$prev"; b="AT_r3_g${g}_s$s"
        d=$(( ${!b:-0} - ${!a:-0} ))
        if [[ -z "$lo" ]] || (( d < lo )); then lo=$d; fi
        if [[ -z "$hi" ]] || (( d > hi )); then hi=$d; fi
      done
      rng="$(( (lo + 30) / 60 ))"
      (( (hi + 30) / 60 != (lo + 30) / 60 )) && rng="$rng-$(( (hi + 30) / 60 ))"
      spacing+="${spacing:+; }seed $prev to seed $s ~$rng min"
    fi
    prev="$s"
  done
  [[ -n "$spacing" ]] && echo "  SEED SPACING at these prices, start to start, for every G: $spacing."
  fu_priced="$(followup_price "$FIRST_G")"
  IFS=$'\t' read -r fu_per fu_n fu_plan <<< "$fu_priced"
  echo "  NOT in that figure: pod boot and checkout, any arm that REFUSES and is re-run,"
  echo "  the seeds a V7 FAIL at seed 0 skips (less), and the follow-up that FAIL prints, run"
  echo "  by hand after the chain: ~$(( (fu_n * fu_per + 59) / 60 )) min a G, $fu_n seeds at --duty $R3_FOLLOWUP_DUTY at $fu_per s each (R3's own"
  echo "  plan at $R3_FOLLOWUP_DUTY, $fu_plan s, plus $R3_RUN_OVERHEAD_S s) (more)."
  echo "on the pod, first:"
  echo "    git -C /workspace/moe-kernels checkout -- moe/bench/hardware/measured_nvidia_h200.yaml"
  echo "    git -C /workspace/moe-kernels fetch origin && git -C /workspace/moe-kernels checkout -B r3-align origin/r3-align"
  echo "    git -C /workspace/moe-kernels log -1 --format=%h    # must print the head that was pushed"
  echo "then detached, APPENDING, so a --resume keeps the earlier passes' console:"
  echo "    cd /workspace/moe-kernels && nohup setsid bash scripts/alpha_g_chain.sh >> /workspace/alpha_g_chain.out 2>&1 < /dev/null &"
  echo "and watch:      tail -f /workspace/alpha_g_chain.out   (and \$SESSION/CHAIN.tsv)"
else
  rebuild_pairs
fi
echo "ledger    $LEDGER"
echo "read a pair on the laptop:  .venv/bin/python scripts/private_weight_reference.py --read RUN1/report.json --replicate-of RUN0/report.json"
echo "  (PAIRS-by-G.tsv already reads every seed of a G together that way; PAIRS-README.txt is the legend)"
YAML_REL="moe/bench/hardware/measured_$CARD.yaml"
CALIB_DIR="$("$PY_BASE" "$HELPERS" calibration-dir "$SESSION/logs/calibrate.log" 2>/dev/null)" || CALIB_DIR=""
CALIB_REL="${CALIB_DIR#"$REPO"/}"
if (( DRY )); then
  CALIB_ARG="-C \"$REPO\" results/calibration/<the run dir calibrate prints>"
elif [[ -n "$CALIB_DIR" && "$CALIB_REL" != "$CALIB_DIR" ]]; then
  CALIB_ARG="-C \"$REPO\" \"$CALIB_REL\""
elif [[ -n "$CALIB_DIR" ]]; then
  CALIB_ARG="-C \"$(dirname "$CALIB_DIR")\" \"$(basename "$CALIB_DIR")\""
else
  CALIB_ARG=""
  echo "  (no '[calibrate] wrote' line in $SESSION/logs/calibrate.log: copy calibrate's run dir"
  echo "   under $REPO/results/calibration off by hand)"
fi
# the console the documented launch appends to, which holds lines no ledger
# does (the lock's fallback notice, the STOP explanations, the per-run lines)
if (( DRY )) || [[ -f "$CONSOLE_OUT" ]]; then
  CONSOLE_ARG="-C \"$(dirname "$CONSOLE_OUT")\" \"$(basename "$CONSOLE_OUT")\""
else
  CONSOLE_ARG=""
  echo "  (no $CONSOLE_OUT: this chain was not launched the documented way, so copy its"
  echo "   console off from wherever it went)"
fi
echo "the ruler this session scored against is $REPO/$YAML_REL, TRACKED and dirty after"
echo "  calibrate: commit it with the results. copy off before releasing the pod:"
echo "    tar czf $WORKSPACE/exfil-alpha_g-$CARD.tar.gz -C \"$(dirname "$SESSION")\" \"$(basename "$SESSION")\" -C \"$(dirname "$RESULTS")\" \"$(basename "$RESULTS")\" -C \"$REPO\" \"$YAML_REL\"${CALIB_ARG:+ $CALIB_ARG}${CONSOLE_ARG:+ $CONSOLE_ARG}"
exit 0
