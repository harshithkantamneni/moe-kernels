#!/usr/bin/env python3
"""How big is an effect this instrument cannot see? The study never asked.

    python scripts/replicate_noise_floor.py --dry-run       # laptop, no GPU
    python scripts/replicate_noise_floor.py --control-only  # laptop, part (b)
    python scripts/replicate_noise_floor.py                 # the pod, both parts
    python scripts/replicate_noise_floor.py --publish       # write the number

WHY THIS EXISTS. Every alpha in this study is a single number from a single run.
Two of them get subtracted and the difference gets a caption. Nowhere does the
study say what the difference between two runs of THE SAME THING would have been,
so no difference it reports has ever been scored against anything. The 2026-09-01
cross-card arm is where that bill came due: A100 minus H200 on 11 matched cells
is +0.0117 in alpha-corrected with a paired sd of 0.0480, and there is no
statement anywhere about whether +0.0117 is a fact about L2 or a fact about
Tuesday. This script produces the missing denominator.

IT HAS TWO PARTS AND THEY ARE INDEPENDENT.

(a) THE REPLICATE FLOOR, which needs a GPU. One arm, run N times identically on
    one card in one session, gives the between-replicate standard deviation of
    alpha at each cell. That is the noise floor: the spread the instrument
    produces when nothing whatsoever has changed. Everything about the design is
    argued below.

(b) THE num_stages CONTROL, which needs no GPU at all and never did. The repo
    already holds two arms that are the SAME CARD at 4 and 3 pipeline stages --
    results/published/2026-09-01-nvidia_h200-alpha-surface-s4 and
    results/published/2026-09-01-nvidia_h200-cross-card-s3 -- and their 11
    matched cells are the SAME 11 the cross-card comparison used. They were
    never differenced. Doing it takes a second and it would have killed the
    cross-card reading on the day: same card, same L2, one scheduling knob moved,
    and alpha moves +0.0101 with sd 0.0323, against +0.0117 with sd 0.0480 for a
    1.5x change in L2 capacity. This part is computed here, published as a
    standing control, and every future cross-card claim is scored against it.

WHAT A REPLICATE IS, and why the unit matters more than the count. Three noise
sources nest inside one another:

  1. timing jitter inside a cell, already summarised by that cell's own median
     (the sweep reports it as timing_spread_median, about 1.4%);
  2. re-running the sweep inside one live process, which reuses the Triton cache,
     the CUDA context, the allocator arenas and the clocks;
  3. re-running the sweep as a FRESH PROCESS with a FRESH Triton cache.

Only (3) is exchangeable with the comparison being scored. A floor measured at
(1) or (2) would be narrower than the noise actually present in every difference
the study reports, and scoring those differences against it is the exact
mechanism by which a null becomes a finding. So a replicate here is a separate
`block_m_crossing_sweep.py` PROCESS with its own output directory, which gives it
its own Triton cache for free -- the sweep already points TRITON_CACHE_DIR at
<out_dir>/triton-cache before it imports vLLM.

WHAT THE PROXY ACTUALLY BOUNDS, corrected 2026-09-02. This docstring said until
today that s3 and s4 "ran in separate processes on separate pods with separate
caches". Separate processes and separate caches are true. SEPARATE PODS IS
FALSE: both arms' ARMS.tsv log to /workspace/session/20260901T214218Z, the same
pod and the same hour, because scripts/cross_card_surface.sh deliberately reuses
the newest session directory. The proxy therefore bounds SAME-SESSION rerun
noise plus num_stages, and nothing about it is established for a comparison
across pods, days or cards -- which is most of what this study subtracts. Part
(a) below inherits the same limit by design (one card, one session), so running
it does not make the cross-pod comparisons scoreable either; saying that plainly
is better than a floor whose scope nobody wrote down.

THE PROXY IS ALSO HETEROSCEDASTIC and must not be quoted per cell. Its 11 paired
deltas split by model into mixtral n=4, sd 0.0486, largest |delta| 0.0888, and
qwen2 n=7, sd 0.0186 -- a 2.61x ratio, inside this file's own
HOMOGENEITY_RATIO of 3.0 but not by much. "Inside 2x the pooled 0.0323" is the
wrong yardstick at mixtral G>=8, where the spread is half again as wide as the
pool. The scope block in NOISE_FLOOR.json carries both numbers.

THE SCOPE OF ANY FLOOR THIS SCRIPT PUBLISHES is written into the JSON and
printed in the plan, because a floor whose scope is only in a docstring gets
quoted without it. It is: one card, one session, the models named in `--arms`,
one BLOCK_SIZE_N, one num_stages, one seed, fresh Triton cache, and the
instrument named by `moe.bench.timing.TIMING_BASIS`. Anything outside that list
is unscored by it.

FRESH CACHE IS THE DEFAULT, and `--warm-cache` measures the other thing on
purpose. Warm is the narrower floor: it holds codegen fixed and reports only
execution variance. It is worth having because the DIFFERENCE between the two
floors is the share of the study's noise that is compilation rather than the
card, but it is never the published floor unless someone asks for it in writing
(`--floor-from warm`), because publishing it would understate every comparison
this repo has ever made.

THE SEED IS HELD FIXED ACROSS REPLICATES, which is a choice with a direction.
Holding it fixed means the weights and the routing histogram are bit-identical in
every replicate, so the floor EXCLUDES data-generation variance. That is correct
here and only here: the comparisons being scored -- s3 vs s4, A100 vs H200 --
also held the seed fixed, so a floor that included data variance would be wider
than their noise and would hide real effects instead of manufacturing them. It
does mean this floor may NOT be used to score any comparison that re-rolled its
inputs.

CHOOSING N, registered before the run rather than after seeing the spread. The
best prior estimate of the per-arm replicate sd available today is the s3-vs-s4
proxy: paired sd 0.03233 over 11 cells, so a per-arm sd of that over sqrt(2),
0.02286. That proxy CONFOUNDS num_stages, so it is an UPPER bound on replicate
noise and N chosen against it is conservative. At that sd, a two-sided 5% test
with 80% power on two conditions of N replicates each at one cell detects

    N =  3   MDE 0.0693        N =  6   MDE 0.0410
    N =  4   MDE 0.0541        N =  8   MDE 0.0344
    N =  5   MDE 0.0462        N = 61   MDE 0.0117

and the floor ESTIMATE itself, pooled over C cells with C(N-1) degrees of
freedom, has an upper 95% confidence bound of

    C=4, N=3  2.87x        C=4, N=5  1.52x
    C=4, N=4  1.65x        C=4, N=6  1.44x

N = 6 is the default because it is the smallest N whose pooled floor over this
script's four default cells is known to better than 1.5x -- a floor quoted to
within a factor of three is not a floor -- and because its 0.0410 detection limit
is 9.4x below the swizzle swing (0.3855) and 10.0x below the G=1 footprint spread
(0.411), the two effects the study most wants to claim. It deliberately does NOT
resolve the cross-card 0.0117: that would need 61 replicates per card at one
cell, and saying so is the point rather than a limitation. It also does not
resolve the QWEN2 swizzle effect (0.0226 at its largest H200 cell), and that is
registered below as an EXPECTED FAIL rather than discovered afterwards.

THE ARMS ARE TWO MODELS, NOT ONE, and that is the whole of the second fix. Until
2026-09-02 DEFAULT_ARMS was mixtral at G=1 and G=16 and nothing else, i.e. the
one model where the swizzle effect is 0.3855 and a 12-sigma certainty. On qwen2
the same lever moves 0.006 to 0.023 on the H200 and reverses sign between tiles
on the A100 (+0.0813 at BLOCK_M=32, -0.0914 at 64) -- below this design's
detection limit, unregistered, and never replicated. A floor measured only where
the effect is largest would have been published as though it licensed the
surface, so `--arms` now covers both models by default and REFUSES a
single-model floor unless `--single-model-floor` says so in argv.

INTERLEAVED, NOT BLOCKED. Until 2026-09-02 the runner ran all six G=1 replicates
and then all six G=16, so C3's swizzle delta was a between-BLOCK difference
carrying whatever drifted across thirteen minutes, while the between-replicate
sd it was scored against was a within-block number. The two swizzles of one
model now run back to back inside one replicate, and the ORDER of the pair
alternates across replicates (`--order counterbalanced`, the default), so a
linear drift inside a pair cancels out of the mean delta instead of being added
to every one of them. `--order paired` keeps G=1 first in every replicate; it is
the weaker design and is kept only because it is the simplest thing to check by
eye. The plan prints the exact sequence on an `order:` line per model.

THE SAME ARITHMETIC ON THE DESIGN THE STUDY ACTUALLY RAN. Paired over k cells
with one replicate per condition and sd_d = 0.0480, the detectable effect at k=11
is 0.0450 (t) or 0.0405 (normal). Detecting 0.0117 needs k = 133 cells. That is
12.0x the cells actually run, which is 3.5x in standard error -- the number the
adversarial evaluation quoted, restated so the units are not ambiguous.

THE SIGN, stated once. Every difference on this page is

    delta = alpha(the arm named SECOND) - alpha(the arm named FIRST)

so `s4 -> s3 delta +0.0101` means THREE stages measures alpha HIGHER than four.
Every printed delta names both arms in that order and no bare signed number
appears without them.

WHAT THIS SCRIPT REFUSES TO DO. It never returns 0.0 for an unmeasured floor.
`noise_floor()` raises `NoiseFloorUnmeasured` until part (a) has run on a real
card, and the published JSON carries `replicate_floor: null` until then, so a
caller that forgets to handle the exception crashes instead of silently deciding
every effect is resolvable. It refuses to pool cells whose spreads disagree by
more than 3x, publishing the widest cell instead. It refuses to write any file
git would ignore. And it refuses to treat a set of replicates whose alphas are
bit-identical as a measurement, because that is what a run-id collision looks
like from the outside and this repo has already shipped one.

EXIT CODES come from `moe.bench.exit_codes` and from nowhere else. This file used
to exit 3 for "nothing was measured" while the session driver read 3 as RETRY and
2 as REFUSED, so every honest refusal was queued for a re-run; and because the
driver's summary grepped free text for `floor|sigma`, a REFUSED log matched
eighteen times and printed the IMPORTED proxy and a PRE-REGISTERED expectation as
though they were this run's measured output. THE WRITING HALF IS FIXED HERE:
refusals return `exit_codes.REFUSED`, a measured run returns
`exit_codes.classify` over its own gates, and the only result-shaped line on the
page is one `exit_codes.result_line` per SCORED gate. A refusal prints none at
all, which is what `classify_text` reads as "nothing was scored".

AND IT READS ITS CHILDREN'S EXIT CODES THROUGH THE SAME TABLE. Every replicate
is a `block_m_crossing_sweep` process, so this file is a CONSUMER of the table as
well as a producer, and until 2026-09-03 it was not: it discarded any child that
exited non-zero without opening its `report.json`. Commit 346b7a5 stopped the
sweep masking a failed CLAIM gate into exit 0, and from that commit a sweep that
measured every cell and merely refuted its own pre-registered claim exited 1 and
was thrown away unread -- four children, four reports on disk, `0 of 4 replicates
ok`, V2 FAIL, this arm INVALID, on the most expensive arm of the session and with
no amount of GPU time able to change it. `REPORT_CODES` is the rule now and both
places that judge a measured child read it.

AND THE CODE IT EXITS WITH NOW GOVERNS THE FILE IT WRITES. `--publish` asked
only whether the run was a rehearsal, so a run whose VALIDITY gates had refused
it printed `exit 3 INVALID: nothing quotable` and wrote
`results/published/NOISE_FLOOR.json` on the way there -- the log and git saying
opposite things about the same run, with the driver invoking this arm WITH
`--publish`. `gates_permit_publishing` is the wall; a failed CLAIM still
publishes, because that is a result about the floor rather than a doubt about it.

AND THAT WALL WAS BUILT AT ONE DOOR OF THREE. `gates_permit_publishing` stands
in the MEASURED publish path. `--control-only --publish` and the `blocked` path
(`--dry-run --publish`, or a real `--publish` run on which `detect_gpu()` comes
back empty) reach `write_published` with a null-floor document and consult no
gate, because neither has a gate to consult. Seeded with a measured floor and
run on 2026-09-03, `--dry-run --publish` printed "NOT A RESULT", returned
REFUSED and left `replicate_floor: null` in git; the driver's real branch passes
`--publish` bare, so one empty CUDA probe on the pod both refused the arm and
deleted a floor an earlier pod had paid 120 minutes for. The rule is a property
of the write now and lives in `write_published`: a document with no floor may
not replace a file that has one.

AND AN UNPLANNED CRASH IS `ERROR`, WHICH IT WAS NOT. `raise SystemExit(main())`
with no handler exits ONE on any exception, and ONE is `CLAIM_FAIL`: a finished
code, latched by the driver, skipped on every resume, RETRY_ARMS 0, session
exits 0. This arm books 120 minutes and pools its children at the very end, so
an OOM in the last minute of it was filed as one of this experiment's registered
findings and never rerun. Measured at 1, on this laptop, by injecting an
exception at the top of `_main`. A string `SystemExit` exits ONE too, and the
two in `write_published` fire AFTER the card is paid for. `main` wraps `_main`
and maps both: unplanned exception to `ERROR`, refusal sentence to `REFUSED`,
sentence to stderr in either case.

THE READING HALF IS NOT FIXED, AND IS NOT THIS FILE'S TO FIX. `arm_gate_regex`
in `scripts/h200_gaps_session.sh` still selects this arm's summary with
`^[[:space:]]*V[0-9][[:space:]]|floor|sigma`, and no wording available to this
file escapes it: a page about a noise floor has to say "floor" and "sigma".
Measured against that regex on 2026-09-02, a REFUSED `--control-only` still
matches 8 lines and a REFUSED `--dry-run` 23, the second including all seven
registered `V1..V7` expectation rows, which is the defect itself. So the shape
changes below are necessary and not sufficient: the defect closes when the
driver greps `^RESULT: ` for this arm, which is the driver's slice to change.
Until it does, a noise_floor block in a session summary is UNVERIFIED unless a
`RESULT:` line stands inside it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe.bench import exit_codes  # noqa: E402
from moe.bench import provenance as PV  # noqa: E402

SWEEP = ROOT / "scripts" / "block_m_crossing_sweep.py"
#: What `block_m_crossing_sweep` names a planted run's directory with, and
#: what it REFUSES a supplied `--run-id` for not beginning with under
#: `--self-test`. Spelled here rather than imported because that module
#: imports at a cost this file pays lazily everywhere else.
SYNTHETIC_DIR_PREFIX = "synthetic-"


def synthetic_run_id(run_id: str, extra: Sequence[str]) -> str:
    """`run_id` under the prefix when `extra` plants a world, unchanged if not.

    A REHEARSAL IS A PLANT AND ITS DIRECTORY HAS TO SAY SO. `sweep_argv` always
    emits `--run-id`, and a supplied id bypasses the sweep's `default_run_id`
    entirely, so until 2026-09-02 a `--rehearse` on the metered machine wrote
    its synthetic `report.json` at the paid replicate's byte-identical path and
    replaced the arm's only machine-readable artefact. The sweep now REFUSES
    that combination rather than silently rewriting the name, because a name it
    rewrote would no longer be the path `rep.report` looks in; taking the prefix
    here is the caller's half of that bargain.

    IT IS A FUNCTION BECAUSE TWO CALLERS NAME THE SAME DIRECTORY. `run_replicate`
    runs it and `render_plan` prints it, and the first fix took the prefix in
    only the run, so every rehearsal id in the printed plan was wrong by exactly
    the prefix that commit added. One decider is what keeps the plan an artefact
    an operator can trust before spending a pod hour.
    """
    return f"{SYNTHETIC_DIR_PREFIX}{run_id}" if "--self-test" in extra else run_id


def timing_basis() -> str | None:
    """`moe.bench.timing.TIMING_BASIS`, or None when it cannot be named here.

    NOT a top-level import, for the same reason `block_m_crossing_sweep` does
    not take one: `moe.bench.timing` imports torch, and `--control-only` and
    `--dry-run` are documented to run on a laptop with no torch at all. None is
    not a default; it says the instrument could not be NAMED on this machine,
    which is exactly the case where nothing was measured either. Broad except
    because a torch that is INSTALLED and broken raises OSError on a missing
    libcudart rather than ImportError, and naming the instrument is never worth
    taking the report down for.
    """
    try:
        from moe.bench.timing import TIMING_BASIS
    except Exception:                                     # noqa: BLE001
        return None
    return TIMING_BASIS


#: The instrument of a document with NO replicate floor in it. Part (b) is
#: arithmetic over committed reports and part (a) did not run, so no apparatus
#: of this repository produced a number on that page and naming the live one
#: would describe a run that never happened. It is a STRING and not None because
#: None reads as "could not be determined", and this is determined: nothing was
#: measured.
UNMEASURED_INSTRUMENT = "not-measured/arithmetic-over-committed-reports"


def unmeasured_provenance() -> PV.Provenance:
    """The block for a page that measured nothing.

    Exists so the two no-GPU `--publish` paths cannot reach `build_document`
    with nothing to stamp. It still records the commit, the dirtiness, the host
    and the packages, which is what makes "this floor was published from an
    unmeasured page at commit X" a checkable statement rather than an absence.
    """
    return PV.provenance_block(instrument=UNMEASURED_INSTRUMENT)


PUBLISHED = ROOT / "results" / "published"

#: Where the importable number lives. `results/*` is ignored with only
#: `!results/published/` excepted, so this is one of the few paths under
#: `results/` git will take, and `git_accepts()` below checks it at write time
#: rather than trusting this comment.
NOISE_FLOOR_JSON = PUBLISHED / "NOISE_FLOOR.json"

SCHEMA = "moe-kernels/noise-floor/2"

#: Every schema `noise_floor()` still knows how to read, oldest first. Version 2
#: (2026-09-02) added `scope`, `prior_sd_by_model`, a provenance block and an
#: instrument, and CORRECTED `prior_sd_source`, which used to say only "upper
#: bound" without saying what it bounded. The additions are additive and v1
#: files still parse, which is why v1 stays in this tuple; the correction is
#: why the version moved at all. A file whose schema is not here is REFUSED
#: rather than parsed on the assumption that the fields mean what they used to.
SCHEMA_READABLE = ("moe-kernels/noise-floor/1", SCHEMA)

#: The two committed arms that are the SAME CARD at different pipeline depths.
#: This pairing is the whole of part (b) and it existed on disk, unread, from the
#: moment the cross-card arm was published.
STAGES_CONTROL_ARMS = (
    PUBLISHED / "2026-09-01-nvidia_h200-alpha-surface-s4",
    PUBLISHED / "2026-09-01-nvidia_h200-cross-card-s3",
)

#: The comparison the control exists to score: two different cards at the SAME
#: pipeline depth. Its reports carry no gpu_name field -- the card is knowable
#: only from the directory name and from `sm_count` -- which is checked, not
#: assumed, by `machine_differs()`.
CROSS_CARD_ARMS = (
    PUBLISHED / "2026-09-01-nvidia_h200-cross-card-s3",
    PUBLISHED / "2026-09-02-nvidia_a100_sxm4_80gb-alpha-surface-s3",
)

#: The field every headline in this study is quoted in. `alpha` is the raw fit
#: and `alpha_upper` is the same fit with the layer's fixed cost removed; all
#: three are differenced because the cross-card result changes SIGN between them
#: and a control that looked at only one would have missed that.
PRIMARY_FIELD = "alpha_corrected"
ALPHA_FIELDS = ("alpha", "alpha_corrected", "alpha_upper")


# --------------------------------------------------------------------------
# the effects this floor exists to score, registered with their sources
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Effect:
    """One difference the study wants to claim, and where its size comes from.

    Written down here so that "is N big enough" is answered against numbers that
    existed before this script ran. An effect added later without a `source` is a
    target moved after the shot.
    """

    name: str
    size: float
    source: str
    #: The model the effect was measured on, or "" when it is not a per-model
    #: quantity. `--arms` is checked against these: an effect registered on a
    #: model no arm measures is a target the run cannot shoot at.
    model: str = ""


EFFECTS: tuple[Effect, ...] = (
    Effect("swizzle swing", 0.3855,
           "s4 arm, mixtral BLOCK_M=32: alpha-corrected 1.0073 at GROUP_SIZE_M=1 "
           "vs 0.6218 at 16", "mixtral-8x7b"),
    # REGISTERED 2026-09-02, and registered because it is SMALL. The surface is
    # published as a general mechanism on the strength of the mixtral number
    # above; on qwen2 the identical lever moves a tenth of that on the H200 and
    # reverses sign between two tiles on the A100, which is what an effect that
    # is not there looks like through a design with an MDE of 0.041. Writing it
    # down before the run is what stops the replicate arm confirming the
    # mechanism on the one model where it is 12 sigma and saying nothing about
    # the other.
    Effect("swizzle swing", 0.0226,
           "H200 s4 and s3, qwen2 G=1 -> G=16 in alpha-corrected: -0.0217 and "
           "-0.0226 (s4, BLOCK_M 32 and 64), -0.0061 and -0.0175 (s3). The "
           "A100 s3 arm gives +0.0813 at BLOCK_M=32 and -0.0914 at 64, i.e. "
           "OPPOSITE SIGNS at matched cells of one arm",
           "qwen2-57b-a14b"),
    Effect("footprint spread at G=1", 0.4110,
           "s4 arm, all seven GROUP_SIZE_M=1 fits: 1.0073 (mixtral BN=64 BM=32) "
           "down to 0.5963 (deepseek-v2-lite BM=32)"),
    Effect("cross-card L2", 0.0117,
           "A100 s3 minus H200 s3, 11 matched cells, paired mean of "
           "alpha-corrected"),
)


def effects_for(model: str) -> tuple[Effect, ...]:
    """Every registered effect that belongs to `model`, plus the model-free ones."""
    return tuple(e for e in EFFECTS if e.model in ("", model))


#: The effect the paired-design power arithmetic is quoted against, looked up by
#: name rather than by position. It used to be `EFFECTS[-1]`, which was correct
#: only while nothing was ever appended to the tuple.
CROSS_CARD_EFFECT = next(e for e in EFFECTS if e.name == "cross-card L2")

def prior_sd() -> float:
    """The best prior estimate of per-arm replicate sd, DERIVED, never re-typed.

    The s3-vs-s4 paired sd of `PRIMARY_FIELD` over 11 matched cells, divided by
    sqrt(2) to turn a difference-of-two sd into a per-arm sd. It CONFOUNDS
    num_stages with rerun noise, so it can only be an upper bound, which is what
    makes it safe to size N against.

    IT USED TO BE THE LITERAL `0.0323 / sqrt(2)` AND THAT LITERAL WAS PUBLISHED.
    `stages_control()` computes the paired sd exactly, 0.03233250623999132, from
    the two committed arms; the constant was the same number re-typed to three
    significant figures, and `build_document` wrote the re-typing into the
    tracked `results/published/NOISE_FLOOR.json` as `prior_sd`
    0.022839549032325487 where the honest value is 0.022862534415054224. Small,
    0.1%, and that is the point: nothing in the file said it was a rounding, the
    number is imported by `scripts/bn_decomposition.py` and by the session
    driver's MDE line, and a published quantity that is a hand-copy of a
    computed one drifts silently the first time the arms it came from move.
    Computed here from those arms, at call time, so it cannot.

    A FUNCTION AND NOT A MODULE CONSTANT because the arms are read off disk, and
    a constant would have to choose between an import that touches
    `results/published` (breaking `--help` on a checkout without it) and a
    literal, which is what it was. Every caller of this already needs the arms.
    It raises whatever `read_arm` raises when they are not there, and refusing
    is the right answer: a plan sized against a fabricated sigma is a plan whose
    N is fiction.
    """
    return stages_control(PRIMARY_FIELD).sd / math.sqrt(2.0)


#: What the proxy is an upper bound ON, spelled out because the old label said
#: only "upper bound" and was read as though it covered every comparison in the
#: repo. Both arms logged to /workspace/session/20260901T214218Z.
PRIOR_SD_SOURCE = (
    "s3-vs-s4 paired sd 0.0323 over 11 cells / sqrt(2). CONFOUNDS num_stages "
    "with rerun noise, so it is an upper bound on SAME-SESSION rerun noise and "
    "on nothing else: both arms ran in the same pod hour "
    "(/workspace/session/20260901T214218Z), so it is NOT established for "
    "cross-pod, cross-day or cross-card comparisons. It is also "
    "heteroscedastic: mixtral n=4 sd 0.0486 against qwen2 n=7 sd 0.0186, a "
    "2.61x ratio, so it must not be quoted per cell at mixtral G>=8")

#: The proxy's spread split by model, from `stages_control()`'s own 11 deltas.
#: Recomputed at run time into the published JSON; these are the values at the
#: commit that registered them, kept so a drift is visible rather than silent.
PRIOR_SD_BY_MODEL = {"mixtral-8x7b": 0.0486, "qwen2-57b-a14b": 0.0186}

DEFAULT_REPLICATES = 6

#: Two-sided 5% at 80% power, the convention every MDE on this page uses. Named
#: rather than inlined so a reader can see that no number here was chosen to make
#: a gate pass.
TEST_LEVEL = 0.05
TEST_POWER = 0.80

#: Cells may only be pooled into one floor if their spreads are comparable. 3x is
#: generous; the failure it prevents is a single wild cell being averaged down
#: into a floor that then declares its own outlier resolvable.
HOMOGENEITY_RATIO = 3.0

#: Identifiable ladder fits expected per arm, used ONLY to size the df column of
#: the pre-run power table. Two, because in the committed s4 arm BLOCK_M 32 and 64
#: carry 33 and 16 memory-bound treads while 128 and 256 carry 0 or 1 and print as
#: not identifiable. It is an expectation; the post-run table uses the count that
#: actually came back.
CELLS_PER_ARM = 2

#: The sweep's own cost model is optimistic because it prices timed calls only.
#: Observed on the s4 arm: ARMS.tsv logged mixtral_g1 at 127 s wall against the
#: model's 54 s for the identical arm. Compiles, allocation and the vLLM import
#: are the difference.
WALL_OVER_MODEL = 127.0 / 54.0


# --------------------------------------------------------------------------
# the sign, in words
# --------------------------------------------------------------------------

def delta_sentence(first: str, second: str, delta: float, unit: str = "") -> str:
    """A signed difference that names both arms in the order it subtracted them.

    Exists because the only thing standing between this study and a reversed
    conclusion is which arm was the minuend, and a table of signed numbers does
    not record that. Every delta printed anywhere below goes through here.
    """
    if delta > 0:
        return f"{second} measures {unit}{delta:+.4f} HIGHER than {first}"
    if delta < 0:
        return f"{second} measures {unit}{delta:+.4f} LOWER than {first}"
    return f"{second} and {first} are exactly equal"


# --------------------------------------------------------------------------
# distributions, so the MDE is a real MDE and not a z-score wearing a costume
# --------------------------------------------------------------------------

_ITMAX, _EPS, _TINY = 400, 3e-16, 1e-300


def _betacf(a: float, b: float, x: float) -> float:
    """Lentz's continued fraction for the incomplete beta. Raises on stall.

    Raises rather than returning its last iterate: a silently unconverged tail
    would move a t quantile by an unknown amount and every MDE on the page is
    that quantile times a standard error.
    """
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    h = d
    for m in range(1, _ITMAX + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        step = d * c
        h *= step
        if abs(step - 1.0) < _EPS:
            return h
    raise ArithmeticError(f"incomplete beta did not converge at a={a}, b={b}, x={x}")


def betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta + a * math.log(x) + b * math.log1p(-x)) * _betacf(a, b, x) / a
    return 1.0 - math.exp(
        lbeta + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: int) -> float:
    if df < 1:
        raise ValueError(f"Student t needs df >= 1, got {df}")
    tail = 0.5 * betai(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


def student_t_ppf(p: float, df: int) -> float:
    """Inverse t by bisection. Exact to ~1e-9 against the printed tables."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {p}")
    lo, hi = -400.0, 400.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _gammp(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a, x), series and CF branches."""
    if x < 0.0 or a <= 0.0:
        raise ValueError(f"P(a, x) needs a > 0 and x >= 0, got a={a}, x={x}")
    if x == 0.0:
        return 0.0
    if x < a + 1.0:
        ap, total, term = a, 1.0 / a, 1.0 / a
        for _ in range(_ITMAX):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * _EPS:
                return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
        raise ArithmeticError(f"incomplete gamma series did not converge at a={a}, x={x}")
    b = x + 1.0 - a
    c = 1.0 / _TINY
    d = 1.0 / b
    h = d
    for i in range(1, _ITMAX):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _TINY:
            d = _TINY
        c = b + an / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        step = d * c
        h *= step
        if abs(step - 1.0) < _EPS:
            return 1.0 - math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    raise ArithmeticError(f"incomplete gamma CF did not converge at a={a}, x={x}")


def normal_ppf(p: float) -> float:
    """Inverse standard normal, by bisection on erfc. Used ONLY where sigma is
    known from outside the two arms being compared, which is exactly what an
    imported noise floor is."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {p}")
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * math.erfc(-mid / math.sqrt(2.0)) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def chi2_ppf(p: float, df: int) -> float:
    if df < 1:
        raise ValueError(f"chi-square needs df >= 1, got {df}")
    lo, hi = 0.0, 4000.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if _gammp(df / 2.0, mid / 2.0) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# power arithmetic -- the API other scripts import
# --------------------------------------------------------------------------

def mde_two_sample(sd: float, n_per_condition: int, *, level: float = TEST_LEVEL,
                   power: float = TEST_POWER) -> float:
    """Smallest difference two arms of n replicates each can resolve.

    The design part (a) runs: two conditions, n independent replicates each, at
    ONE cell, compared with a two-sample t test.
    """
    if n_per_condition < 2:
        raise ValueError("a two-sample MDE needs at least 2 replicates per condition")
    if sd <= 0:
        raise ValueError(f"MDE needs a positive sd, got {sd}")
    df = 2 * n_per_condition - 2
    factor = student_t_ppf(1.0 - level / 2.0, df) + student_t_ppf(power, df)
    return factor * sd * math.sqrt(2.0 / n_per_condition)


def mde_at_measured_floor(sd: float | None, n: int, *, external: bool
                          ) -> float | None:
    """An MDE at the floor THIS RUN measured, or None when there is no floor.

    A pooled sd of exactly zero is not a very good floor, it is V3's collision:
    every replicate of a cell came back with the same bits. `mde_two_sample`
    and `mde_external_sigma` both RAISE on a non-positive sd, correctly, because
    a detection limit of zero would say every difference is resolvable.

    WHAT THAT COST BEFORE THIS EXISTED. C2 and C3 called those two functions
    directly on the measured floor, so a run V3 had already failed crashed
    inside `claim_gates` before a single gate was printed. The process then
    exits 1 -- there is no handler above it -- and 1 is CLAIM_FAIL, which the
    session driver's ledger latches as finished and skips on every resume. A
    degenerate arm would have been recorded as a refuted claim and never rerun.
    The arm has to reach INVALID with V3 naming the collision in words, so these
    gates report UNKNOWN rather than raising, and UNKNOWN is already scored
    against a CLAIM gate by `exit_codes.classify`.

    TWO CALL SITES, C2 AND C3, and fixing either alone leaves the crash: C3
    runs after C2 on the same degenerate floor.
    """
    if sd is None or sd <= 0:
        return None
    if n < (1 if external else 2):
        return None
    return mde_external_sigma(sd, n) if external else mde_two_sample(sd, n)


def mde_external_sigma(sigma: float, n_per_condition: int = 1, *,
                       level: float = TEST_LEVEL, power: float = TEST_POWER) -> float:
    """Smallest difference resolvable when sigma comes from an IMPORTED floor.

    THE ONLY FORMULA THAT DESCRIBES THIS STUDY'S OWN DESIGN. Every published
    difference in this repo is one run per condition, so there is no within-arm
    variance estimate and no two-sample t test exists: `mde_two_sample` cannot be
    evaluated at n=1 because its df is zero. With sigma supplied from the
    replicate floor the test is a known-variance z test and n=1 is legitimate:
    it costs (1.960 + 0.842) * sqrt(2) = 3.96 sigma.

    IT IS ALSO THE STRICTER CHOICE, which is why gate C2 uses it. The tempting
    substitution -- quote `mde_two_sample` at n=2 and call it close enough -- is
    5.36 sigma, because at 2 degrees of freedom the t quantile is 4.303 rather
    than 1.960. That is 35% LOOSER, and C2's claim is that an observed difference
    sits BELOW the limit, so the loose number would have made C2 easier to pass.
    Using 3.96 sigma instead means C2 has to clear the harder bar. At the prior
    sigma of 0.02286 the limit is 0.0905, still 7.7x the cross-card difference
    the study reported.
    """
    if n_per_condition < 1:
        raise ValueError("need at least one run per condition")
    if sigma <= 0:
        raise ValueError(f"MDE needs a positive sigma, got {sigma}")
    z = normal_ppf(1.0 - level / 2.0) + normal_ppf(power)
    return z * sigma * math.sqrt(2.0 / n_per_condition)


def mde_paired(sd_of_differences: float, cells: int, *, level: float = TEST_LEVEL,
               power: float = TEST_POWER) -> float:
    """Smallest mean paired difference k matched cells can resolve.

    The design the STUDY ran: one replicate per condition, paired across cells,
    with the spread coming from cell-to-cell disagreement rather than from
    reruns. Named separately from `mde_two_sample` because the two answer
    different questions and quoting one for the other is how "underpowered by
    3.5x" turned into an argument about whether it meant 3.5 or 12.
    """
    if cells < 2:
        raise ValueError("a paired MDE needs at least 2 cells")
    if sd_of_differences <= 0:
        raise ValueError(f"MDE needs a positive sd, got {sd_of_differences}")
    df = cells - 1
    factor = student_t_ppf(1.0 - level / 2.0, df) + student_t_ppf(power, df)
    return factor * sd_of_differences / math.sqrt(cells)


def replicates_for(effect: float, sd: float, *, cap: int = 500, **kw) -> int | None:
    """Smallest n per condition whose two-sample MDE reaches `effect`, or None.

    None rather than `cap` when the effect is out of reach inside the cap, so a
    caller cannot mistake "500 would do it" for "500 is what it takes".
    """
    for n in range(2, cap + 1):
        if mde_two_sample(sd, n, **kw) <= effect:
            return n
    return None


def cells_for(effect: float, sd_of_differences: float, *, cap: int = 5000,
              **kw) -> int | None:
    """Smallest k whose paired MDE reaches `effect`, or None inside the cap."""
    for k in range(2, cap + 1):
        if mde_paired(sd_of_differences, k, **kw) <= effect:
            return k
    return None


def sd_upper_bound(sd: float, df: int, *, level: float = TEST_LEVEL) -> float:
    """Upper (1 - level) confidence bound on a pooled sd with `df` degrees.

    A floor quoted without this is a floor quoted to an unknown factor: at 4 df
    the true sd can be 2.9x the estimate and every "below the noise" verdict
    drawn from it is worthless.
    """
    if df < 1:
        raise ValueError(f"an sd bound needs df >= 1, got {df}")
    return sd * math.sqrt(df / chi2_ppf(level / 2.0, df))


# --------------------------------------------------------------------------
# the importable number, and its refusal to be absent quietly
# --------------------------------------------------------------------------

class NoiseFloorUnmeasured(RuntimeError):
    """Raised instead of returning a number nobody measured."""


class EffectBelowNoiseFloor(AssertionError):
    """Raised when a caller tries to claim an effect the instrument cannot see."""


@dataclass(frozen=True)
class Floor:
    """A measured replicate floor and everything needed to judge whether it
    applies to the comparison a caller wants to score."""

    sd: float
    df: int
    upper95: float
    field: str
    n_replicates: int
    cells: int
    cache_mode: str
    gpu_name: str
    pooled: bool
    provenance: str

    def mde(self, n_per_condition: int = 1) -> float:
        """The detection limit this floor imposes at n runs per condition."""
        return mde_external_sigma(self.sd, n_per_condition)

    def resolves(self, effect: float, *, n_per_condition: int = 1) -> bool:
        """Could a comparison at this floor have seen an effect that size?

        `n_per_condition=1` is the study's own design -- one run per arm -- and
        is the default because that is what every published difference in this
        repo actually is. Sigma comes from this floor rather than from the two
        arms, so the known-variance form is the right one; see
        `mde_external_sigma` for why the two-sample form must not be substituted.
        """
        return abs(effect) >= self.mde(n_per_condition)


def noise_floor(path: Path | None = None, field_name: str = PRIMARY_FIELD,
                *, allow_synthetic: bool = False) -> Floor:
    """The measured floor, or an exception. Never a default.

        from replicate_noise_floor import noise_floor, EffectBelowNoiseFloor
        floor = noise_floor()
        if not floor.resolves(0.0117):
            raise EffectBelowNoiseFloor(...)

    Raises `NoiseFloorUnmeasured` when the file is absent, when part (a) has not
    run (`replicate_floor` is null), when the stored floor is synthetic, or when
    the requested field was not measured. Returning a placeholder here would let
    every caller silently conclude that every effect is resolvable, which is the
    state the study is in today and the reason this file exists.
    """
    target = path or NOISE_FLOOR_JSON
    if not target.exists():
        raise NoiseFloorUnmeasured(
            f"{target} does not exist. Run scripts/replicate_noise_floor.py on a "
            f"GPU and then --publish. There is no default noise floor and there "
            f"will not be one.")
    doc = json.loads(target.read_text())
    schema = doc.get("schema")
    if schema not in SCHEMA_READABLE:
        raise NoiseFloorUnmeasured(
            f"{target} carries schema {schema!r}, which this module does not "
            f"know how to read (it reads {list(SCHEMA_READABLE)}). Parsing it "
            f"anyway would mean assuming its fields still mean what they meant "
            f"here, and the one field that has already changed meaning is the "
            f"scope of the prior.")
    block = doc.get("replicate_floor")
    if not block:
        raise NoiseFloorUnmeasured(
            f"{target} carries stages_control but replicate_floor is null: part "
            f"(a) has not run on a card yet. The num_stages control is available "
            f"through stages_control() and is NOT a substitute -- it confounds "
            f"pipeline depth with rerun noise.")
    if block.get("synthetic") and not allow_synthetic:
        raise NoiseFloorUnmeasured(
            f"{target} holds a REHEARSAL floor generated from the sweep's own "
            f"model, not a measurement. Pass allow_synthetic=True only to test "
            f"the plumbing.")
    per_field = block.get("per_field", {})
    if field_name not in per_field:
        raise NoiseFloorUnmeasured(
            f"{target} has no floor for {field_name!r}; measured fields are "
            f"{sorted(per_field)}")
    entry = per_field[field_name]
    return Floor(sd=entry["sd"], df=entry["df"], upper95=entry["upper95"],
                 field=field_name, n_replicates=block["n_replicates"],
                 cells=entry["cells"], cache_mode=block["cache_mode"],
                 gpu_name=block["gpu_name"], pooled=entry["pooled"],
                 provenance=block["provenance"])


def sizing_sigma(path: Path | None = None,
                 field_name: str = PRIMARY_FIELD) -> tuple[float, str, str]:
    """`(sigma, "MEASURED" or "ASSUMED", where it came from)`, for a caller that
    must print a detection limit whether or not the card has run yet.

    `noise_floor()` RAISES while part (a) has not run, and that is the right
    answer for a caller that must not proceed without a measurement. It is the
    wrong answer for the three that must print SOMETHING every session: they
    need the measured floor when one exists and the declared prior, LABELLED,
    when it does not. `scripts/h200_gaps_session.sh:mde_line` writes exactly
    that fallback out longhand and gets it right. The other two consumers,
    `scripts/alpha_surface.py:prior_sd` and
    `scripts/bn_decomposition.py:published_prior_sd`, do not have it at all:
    they read `payload["prior_sd"]` directly, which is the s3/s4 proxy, and
    they will still be reading the proxy after a perfect pod run publishes a
    measured floor into the same file. That is the arm buying a tracked number
    two of its three consumers cannot see.

    ONE FALLBACK, HERE, rather than three copies of it in three files, because
    three copies of one rule is the shape this rebuild has now found seven
    times. The basis word is returned rather than left to the caller to
    compose: a sigma printed without it reads as a measurement, and for most of
    this study's life it will not be one.

    It never invents a number. When the file is unreadable, carries no usable
    prior and holds no floor, it raises `NoiseFloorUnmeasured` like every other
    reader here.
    """
    target = path or NOISE_FLOOR_JSON
    try:
        floor = noise_floor(target, field_name)
    except NoiseFloorUnmeasured as exc:
        # The first SENTENCE, split on ". " and not on ".", because every one of
        # these messages opens with the path and the path has a `.json` in it.
        unmeasured = " ".join(str(exc).split()).split(". ")[0]
    else:
        return floor.sd, "MEASURED", floor.provenance
    try:
        doc = json.loads(target.read_text())
        prior = float(doc["prior_sd"])
        source = doc.get("prior_sd_source", "unsourced")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise NoiseFloorUnmeasured(
            f"{target} carries no measured floor ({unmeasured}) and no readable "
            f"prior_sd either ({type(exc).__name__}), so there is no sigma to "
            f"size against. Publishing part (b) is one command and needs no "
            f"GPU: replicate_noise_floor.py --control-only --publish") from exc
    if not prior > 0:
        raise NoiseFloorUnmeasured(
            f"{target} carries prior_sd {prior!r}, which is not a spread.")
    return prior, "ASSUMED", f"{source} [no measured floor: {unmeasured}]"


def assert_resolvable(effect: float, label: str, *, n_per_condition: int = 1,
                      path: Path | None = None,
                      field_name: str = PRIMARY_FIELD) -> Floor:
    """Refuse to let a claim smaller than the instrument through.

    The one-line call any future cross-card, cross-dtype or cross-swizzle claim
    should make before it is written down.
    """
    floor = noise_floor(path, field_name)
    if not floor.resolves(effect, n_per_condition=n_per_condition):
        raise EffectBelowNoiseFloor(
            f"{label}: |{effect:+.4f}| is below what {n_per_condition} run(s) per "
            f"condition can resolve at the measured floor sd={floor.sd:.4f} "
            f"(MDE {floor.mde(n_per_condition):.4f}). "
            f"This is not a null result; it is an unresolvable measurement.")
    return floor


# --------------------------------------------------------------------------
# part (b): reading committed reports and differencing matched cells
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LadderCell:
    """One identifiable ladder fit, keyed by everything that was held fixed."""

    model: str
    dtype: str
    group_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    block_m: int
    values: dict[str, float | None]
    treads: int
    sm_count: int
    source: str

    @property
    def key(self) -> tuple:
        """Everything EXCEPT num_stages and the card, which are the two things a
        control and a cross-card comparison respectively vary."""
        return (self.model, self.dtype, self.group_m, self.block_n,
                self.block_k, self.num_warps, self.block_m)


def read_arm(directory: Path) -> list[LadderCell]:
    """Every ladder fit in one published arm.

    Raises on a directory with no reports rather than returning an empty list: an
    empty arm silently produces an empty intersection, which prints as a clean
    "0 matched cells" and looks like a finding about the data.
    """
    reports = sorted(directory.glob("*.report.json"))
    if not reports:
        raise FileNotFoundError(f"no *.report.json under {directory}")
    cells: list[LadderCell] = []
    for path in reports:
        doc = json.loads(path.read_text())
        fixed = doc["fixed"]
        for block_m, fit in doc["ladder"].items():
            cells.append(LadderCell(
                model=doc["model"], dtype=doc["dtype"],
                group_m=fixed["GROUP_SIZE_M"], block_n=fixed["BLOCK_SIZE_N"],
                block_k=fixed["BLOCK_SIZE_K"], num_warps=fixed["num_warps"],
                num_stages=fixed["num_stages"], block_m=int(block_m),
                values={f: fit.get(f) for f in ALPHA_FIELDS},
                treads=fit.get("memory_points", 0),
                sm_count=doc.get("sm_count", 0),
                source=f"{directory.name}/{path.name}"))
    return cells


@dataclass(frozen=True)
class PairedDifference:
    """A paired comparison of two arms over their matched, identifiable cells."""

    first: str
    second: str
    field: str
    pairs: tuple[tuple[tuple, float, float], ...]
    varied: tuple[str, ...]
    same_machine: bool | None

    @property
    def deltas(self) -> list[float]:
        return [b - a for _, a, b in self.pairs]

    @property
    def n(self) -> int:
        return len(self.pairs)

    @property
    def mean(self) -> float | None:
        return statistics.fmean(self.deltas) if self.pairs else None

    @property
    def sd(self) -> float | None:
        return statistics.stdev(self.deltas) if self.n >= 2 else None

    @property
    def mde(self) -> float | None:
        sd = self.sd
        return mde_paired(sd, self.n) if sd and sd > 0 and self.n >= 2 else None

    @property
    def resolved(self) -> bool | None:
        """Is |mean| above what this design could detect? None when unknowable."""
        m, d = self.mean, self.mde
        return None if m is None or d is None else abs(m) >= d

    def line(self) -> str:
        if self.mean is None:
            return f"{self.first} -> {self.second} [{self.field}]: no matched cells"
        sd = self.sd
        mde = self.mde
        verdict = {True: "RESOLVED", False: "BELOW THE DETECTION LIMIT",
                   None: "UNKNOWN"}[self.resolved]
        return (f"{self.field:16s} n={self.n:2d}  "
                f"{delta_sentence(self.first, self.second, self.mean)}  "
                f"sd {sd:.4f}" + (f"  MDE {mde:.4f}" if mde else "")
                + f"  -> {verdict}")


def pair_arms(first_dir: Path, second_dir: Path, field_name: str,
              ) -> PairedDifference:
    """Match two arms cell by cell and difference the requested alpha field.

    Cells match on model, dtype, GROUP_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    num_warps and BLOCK_M -- everything the sweep pins except num_stages and the
    card. A cell where either side is not identifiable is DROPPED, and the count
    of drops is recoverable from `n` against the arm sizes; a fit that is not
    identifiable is not a zero and must never be differenced as one.
    """
    a_cells = {c.key: c for c in read_arm(first_dir)}
    b_cells = {c.key: c for c in read_arm(second_dir)}
    pairs = []
    for key in sorted(set(a_cells) & set(b_cells), key=repr):
        a, b = a_cells[key], b_cells[key]
        av, bv = a.values.get(field_name), b.values.get(field_name)
        if av is None or bv is None:
            continue
        pairs.append((key, av, bv))
    varied = []
    a_any = next(iter(a_cells.values()))
    b_any = next(iter(b_cells.values()))
    if a_any.num_stages != b_any.num_stages:
        varied.append("num_stages")
    same_machine = machine_differs(a_cells.values(), b_cells.values())
    if same_machine is False:
        varied.append("gpu")
    return PairedDifference(first=first_dir.name, second=second_dir.name,
                            field=field_name, pairs=tuple(pairs),
                            varied=tuple(varied), same_machine=same_machine)


def machine_differs(a_cells, b_cells) -> bool | None:
    """Same card or not, decided by SM count, which the report DOES record.

    The report JSON carries no gpu_name, so the only in-band evidence of which
    card produced it is `sm_count` -- 132 on the H200, 108 on the A100. Returns
    True for same machine, False for different, and None when either side did not
    record it, because "the directory is called a100" is a filename and not a
    measurement.
    """
    a_sm = {c.sm_count for c in a_cells if c.sm_count}
    b_sm = {c.sm_count for c in b_cells if c.sm_count}
    if not a_sm or not b_sm:
        return None
    if len(a_sm) > 1 or len(b_sm) > 1:
        return None
    return a_sm == b_sm


def stages_control(field_name: str = PRIMARY_FIELD) -> PairedDifference:
    """The standing control: one card, one L2, num_stages 4 -> 3.

    THE NUMBER ANY CROSS-CARD CLAIM IS SCORED AGAINST. If moving a scheduling
    knob on one machine moves alpha as much as moving to another machine does,
    the cross-card difference is not evidence about L2.
    """
    return pair_arms(*STAGES_CONTROL_ARMS, field_name)


def cross_card(field_name: str = PRIMARY_FIELD) -> PairedDifference:
    """The comparison under scrutiny: two cards, same pipeline depth."""
    return pair_arms(*CROSS_CARD_ARMS, field_name)


# --------------------------------------------------------------------------
# part (a): the arms, and a run id that cannot collide
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Arm:
    """One sweep configuration, replicated N times.

    The defaults are the two ends of the swizzle swing, on BOTH models the s3/s4
    surface was fitted on, so the same replicates that measure the floor score
    the largest effect the study claims AND the one it never registered.

    THE TIMING FIELDS ARE THE INSTRUMENT'S, NOT A CALL COUNT. `--warmup` on the
    sweep became MILLISECONDS of delivered load on 2026-09-02 and `--iters` was
    retired as a timing knob (`moe.bench.timing.time_kernel` sizes the count per
    cell from `--cell-budget-ms`). This arm carried `warmup=20` from before that
    change, which the new parser reads as 20 MILLISECONDS -- a fifteenth of the
    sweep's own 300 ms default, so every replicate would have been timed at an
    unsettled clock and the floor would have measured the governor. `warmup_ms`,
    `trials` and `l2_flush` are the instrument's three knobs and they are passed
    through by name.

    BLOCK_M=128 IS NOT SWEPT HERE AND ITS ABSENCE PAYS FOR THE SECOND MODEL.
    The floor is a spread of FITTED alphas, so a tile with no fit contributes
    nothing to it, and 128 has no fit anywhere in the committed s3/s4 arms at
    these settings: `ladder["128"]["alpha_corrected"]` is null in every mixtral
    and qwen2 report at BLOCK_SIZE_N=64, because at this geometry its cap sits
    on the ridge and `fit_ladder` discards a memory branch within 15% of the
    compute branch. 256 stays because it is not a subject either: it is the
    COMPUTE REFERENCE, `compute_reference` qualifies the LARGEST ladder and
    refuses rather than falling through to the runner-up, so dropping it would
    leave 64 as the largest and cost the run every alpha it has. Three tiles
    instead of four is 256 s of modelled GPU per arm against 413, which is what
    makes a two-model floor cost about what the one-model floor cost.
    """

    name: str
    model: str = "mixtral-8x7b"
    dtype: str = "bf16"
    group_m: int = 1
    block_n: int = 64
    num_stages: int = 4
    tiles: str = "32,64,256"
    r_max: int = 1024
    row_step: int = 32
    step_probes: int = 6
    warmup_ms: float = 300.0
    trials: int = 3
    l2_flush: bool = True
    cell_budget_ms: float = 400.0
    seed: int = 0

    @property
    def swizzle_label(self) -> str:
        """`g1`, `g16`: the token this arm shows up as on the `order:` line."""
        return f"g{self.group_m}"

    def sweep_argv(self, run_id: str, out_dir: Path) -> list[str]:
        argv = [
            "--model", self.model, "--dtype", self.dtype, "--tiles", self.tiles,
            "--r-max", str(self.r_max), "--row-step", str(self.row_step),
            "--step-probes", str(self.step_probes),
            "--num-stages", str(self.num_stages), "--group-m", str(self.group_m),
            "--block-n", str(self.block_n), "--warmup", str(self.warmup_ms),
            "--trials", str(self.trials),
            "--cell-budget-ms", str(self.cell_budget_ms), "--seed", str(self.seed),
            "--run-id", run_id, "--out", str(out_dir),
        ]
        if not self.l2_flush:
            argv.append("--no-l2-flush")
        return argv


MIXTRAL, QWEN2 = "mixtral-8x7b", "qwen2-57b-a14b"

DEFAULT_ARMS: tuple[Arm, ...] = (
    Arm("mixtral_g1", model=MIXTRAL, group_m=1),
    Arm("mixtral_g16", model=MIXTRAL, group_m=16),
    Arm("qwen2_g1", model=QWEN2, group_m=1),
    Arm("qwen2_g16", model=QWEN2, group_m=16),
)

#: Every arm, in the order `--arms` names them by default. Both models, both
#: ends of the swizzle: see the module docstring for why one model is not a
#: floor.
DEFAULT_ARM_NAMES = ",".join(a.name for a in DEFAULT_ARMS)

ORDER_COUNTERBALANCED = "counterbalanced"
ORDER_PAIRED = "paired"
ORDER_MODES = (ORDER_COUNTERBALANCED, ORDER_PAIRED)


def swizzle_pairs(arms: list[Arm]) -> dict[str, list[Arm]]:
    """`{model: [arms, sorted by GROUP_SIZE_M]}`, which is what gets interleaved.

    Grouped by MODEL because the contrast the interleave protects is within a
    model: a mixtral G=16 run beside a qwen2 G=1 run is not a pair of anything.
    """
    by: dict[str, list[Arm]] = {}
    for arm in arms:
        by.setdefault(arm.model, []).append(arm)
    return {m: sorted(v, key=lambda a: a.group_m) for m, v in sorted(by.items())}


def run_order(arms: list[Arm], n: int, mode: str = ORDER_COUNTERBALANCED
              ) -> list[tuple[Arm, int]]:
    """The launch sequence: `[(arm, replicate index), ...]`, interleaved.

    THE DEFECT THIS REPLACES ran `for arm in arms: for index in 1..n`, i.e. all
    six G=1 replicates and then all six G=16. The between-replicate sd it
    published was then a within-block number while the swizzle delta it scored
    against that sd was a between-block difference, carrying every drift that
    happened across the thirteen minutes between the blocks. The repo's own
    record of how big that can be is the H200 dense peak moving 7.1% between
    sessions.

    COUNTERBALANCED is the default and reverses the pair's order on alternate
    replicates (g1,g16 / g16,g1 / g1,g16 / ...). A linear drift inside a pair
    then contributes +d to half the deltas and -d to the other half and cancels
    out of the mean, at the cost of a slightly WIDER per-arm replicate sd --
    which is the conservative direction for a floor. PAIRED keeps g1 first in
    every replicate: every contrast is still within minutes, but a within-pair
    drift biases every delta the same way and survives averaging.

    Models are cycled outermost so the two models' pairs interleave too, which
    keeps neither model's replicates bunched at one end of the session.
    """
    if mode not in ORDER_MODES:
        raise ValueError(f"unknown order {mode!r}; known modes are {ORDER_MODES}")
    pairs = swizzle_pairs(arms)
    out: list[tuple[Arm, int]] = []
    for index in range(1, n + 1):
        for _model, group in pairs.items():
            ordered = group
            if mode == ORDER_COUNTERBALANCED and index % 2 == 0:
                ordered = list(reversed(group))
            out += [(arm, index) for arm in ordered]
    return out


def order_lines(arms: list[Arm], n: int, mode: str) -> list[str]:
    """One `order:` line per model, naming the exact launch sequence.

    Greppable on purpose and it is the acceptance check for the interleave:

        --dry-run | grep -E 'order: [^ ]+ (g1,g16,g16,g1,){2}g1,g16,g16,g1'
        --order paired --dry-run | grep -E 'order: [^ ]+ (g1,g16,){5}'

    The pattern is printed rather than described because a description of an
    order is not an order, and the blocked design this replaces was described
    as interleaved in its own predictions text.
    """
    seq: dict[str, list[str]] = {}
    for arm, _index in run_order(arms, n, mode):
        seq.setdefault(arm.model, []).append(arm.swizzle_label)
    return [f"order: {model} {','.join(labels)}   [{mode}]"
            for model, labels in seq.items()]


def run_id_for(arm: Arm, replicate: int, *, gpu_name: str, cache_mode: str,
               sweep_args: Sequence[str], order: str, python: str) -> str:
    """A run id carrying EVERY swept parameter, the card, and the replicate index.

    `block_m_crossing_sweep.default_run_id` OMITTED THE GPU until 2026-09-02,
    and that omission was not hypothetical: the A100 cross-card arm and the H200
    cross-card arm are committed under IDENTICAL filenames
    (`mixtral-8x7b-bf16-r1024-g1-n64-4867a2.report.json` in both), so only the
    directory name distinguishes two different machines. It takes the card now,
    so the card here is belt and braces rather than the workaround it was.

    THE REPLICATE INDEX IS STILL OURS AND ALWAYS WILL BE: the sweep has no
    notion of one. Six replicates of an arm would otherwise derive six identical
    ids, resume into one directory, find every cell already on disk, skip all of
    them and report replicate 1's timings six times with a between-replicate sd
    of exactly 0.0000. The V3 gate below exists to catch that even if this
    function is wrong.

    THE ID IS BUILT BY `moe.bench.provenance.run_id` AS OF 2026-09-02 and not by
    a private hash here, so the card is a REQUIRED keyword that raises `NoCard`
    when absent and every knob is refused when it is None. Three scripts had
    each re-implemented a subset of this and each had left a different knob out.

    `sweep_args` AND `order` ARE REQUIRED KEYWORDS, ADDED 2026-09-02, and they
    are required rather than defaulted because a default is what a new call site
    forgets. Both were absent and both were the lost-arm collision again:

      * `--sweep-arg` is a PASSTHROUGH: whatever it carries is appended to every
        child `block_m_crossing_sweep` command line, so
        `--sweep-arg=--group-m --sweep-arg=16` re-swept the G=16 arm under the
        G=1 id. The child cannot rescue itself, because this parent hands it
        `--run-id` (see `Arm.sweep_argv`) and the child then names its directory
        after ours. On a pod the G=16 arm would have resumed the G=1 directory,
        found every cell present, spent no GPU time, and produced a noise floor
        computed from G=1 timings labelled G=16.
      * `--order` decides the counterbalancing DESIGN. `counterbalanced` and
        `paired` produced byte-identical id sets, so switching the design
        resumed the other design's cells and the drift cancellation this
        module's docstring argues for at length would never have happened. It
        is a design knob, not an analysis knob.

    The passthrough enters the key as the operator SPELLED it, not parsed: this
    function has no business knowing the sweep's grammar, and two spellings of
    one setting landing in two directories is the safe direction of that
    ignorance.

    `python` IS REQUIRED TOO, ADDED 2026-09-02 AND FOR THE SAME REASON. It is
    the interpreter every child sweep is launched with, so it selects the venv
    and therefore the torch, triton and vLLM that measure every cell, and the
    session driver deliberately varies it between `PY_BASE` and `PY_VLLM` from
    one arm to the next. Two interpreters resumed one directory:

        --dry-run --gpu-name 'NVIDIA H200' --python /usr/bin/python3
        --dry-run --gpu-name 'NVIDIA H200' --python /other/venv/bin/python

    printed the identical `...-settingscell_budget-fbf9bcea`. It travels under
    `settings` rather than in the visible part because the path is long and the
    six knobs an operator reads in `ls` are worth more room than it is; the hash
    does not care where a knob sits.

    THIS WAS FOUND BY A COMPLETENESS GUARD, NOT BY A READER.
    `tests/test_replicate_noise_floor.py` now reads the knobs off `build_parser`
    and demands each one be exercised or refused by name, which is the same
    guard `tests/test_run_ids.py` grew after a hand-written list of ten knobs
    against a parser of sixteen let `--cell-budget-ms` through.
    """
    return PV.run_id(
        card=gpu_name,
        # `run_id` renders the knobs in NAME ORDER and truncates the visible
        # part at 96 characters, so the six knobs a human needs in `ls` are
        # named to sort ahead of the rest and the rest travel under one
        # `settings` key. Every one of them still enters the hash, which is
        # what actually keeps two replicates apart; the ordering only decides
        # what survives the truncation.
        cache=cache_mode,
        g=arm.group_m,
        model=arm.model,
        n=arm.block_n,
        order=order,
        rep=replicate,
        s=arm.num_stages,
        settings={
            "sweep_args": list(sweep_args),
            "python": python,
            "dtype": arm.dtype,
            "tiles": arm.tiles,
            "r_max": arm.r_max,
            "row_step": arm.row_step,
            "step_probes": arm.step_probes,
            "warmup_ms": arm.warmup_ms,
            "trials": arm.trials,
            "l2_flush": arm.l2_flush,
            "cell_budget_ms": arm.cell_budget_ms,
            "seed": arm.seed,
        },
    )


#: The exit codes a MEASURED child sweep comes back with, and the only ones
#: after which this parent opens its `report.json`. The plan-side twin is
#: `PLAN_CODES`, which answers the same question for a `--dry-run` cost probe.
#:
#: THE PARENT USED TO KEY ON `returncode == 0`, IN TWO PLACES, AND EITHER ONE
#: ALONE THREW THE ARM AWAY. `run_replicate` set `rep.error` on any non-zero
#: code without opening the report, and `Replicate.ok` demanded a literal 0 a
#: second time, so a fix applied to one of them changed nothing. While
#: `block_m_crossing_sweep` masked a failed CLAIM gate into exit 0 the coupling
#: was invisible; commit 346b7a5 stopped the masking, and from that commit a
#: sweep that measured every cell perfectly well and merely refuted its own
#: pre-registered claim exited 1 and had every cell discarded unread.
#: Reproduced off GPU at `--rehearse 0.4 --replicates 2`: four children, four
#: `report.json` on disk, `0 of 4 replicates ok`, V2 FAIL, exit INVALID. On the
#: pod that is two hours of a rented card for a guaranteed-useless result, and
#: an INVALID arm is latched by the session driver's ledger and skipped on
#: every resume, so it cannot even be retried without hand-editing the ledger.
#:
#: WHY CLAIM_FAIL IS IN HERE AND INVALID IS NOT, though `exit_codes` puts both
#: in `MEASURED_CODES`. That set answers "do this arm's cells exist on disk",
#: which decides whether a directory is worth keeping. This one answers "may
#: this parent POOL those cells into a floor", which is a different question
#: with a different answer: an INVALID child is one whose own VALIDITY gate
#: failed AFTER measuring, so nothing on its page may be quoted, while a
#: CLAIM_FAIL child measured soundly and the world disagreed with it. A floor
#: is a spread, not a claim, and the spread is the one thing this parent wants
#: from a child regardless of what the child concluded. REFUSED measured
#: nothing and ERROR crashed, so neither leaves a report to read.
REPORT_CODES = (exit_codes.DONE, exit_codes.CLAIM_FAIL)


def code_census(replicates: Sequence[Replicate]) -> str:
    """`"2 DONE, 2 CLAIM_FAIL"`, in table order, for the V2 line.

    A floor pooled over four children of which two refuted their own claim is
    not the same artefact as one pooled over four clean children, and the
    difference must be visible in the one line the session driver greps rather
    than only in four child logs nobody opens on a metered pod.
    """
    if not replicates:
        return "nothing launched"
    counts: dict[str, int] = {}
    for rep in replicates:
        name = ("no exit code" if rep.returncode is None
                else exit_codes.CODE_NAMES.get(rep.returncode,
                                               f"off-table {rep.returncode}"))
        counts[name] = counts.get(name, 0) + 1
    order = list(exit_codes.CODE_NAMES.values())
    return ", ".join(
        f"{counts[k]} {k}" for k in
        sorted(counts, key=lambda k: (order.index(k) if k in order else 99, k)))


@dataclass
class Replicate:
    """One completed (or failed) sweep process."""

    arm: str
    index: int
    run_id: str
    out_dir: Path
    report: Path
    returncode: int | None = None
    seconds: float = 0.0
    error: str = ""
    cells: list[LadderCell] = field(default_factory=list)
    compiles: dict[int, int] = field(default_factory=dict)
    #: `moe.bench.timing.TIMING_BASIS` as the sweep stamped it into this
    #: replicate's report. "" means the report carried no instrument at all,
    #: which is a report from before 2026-09-02 and is NOT the same state as a
    #: report timed by a different named instrument. V6 tells the two apart.
    instrument: str = ""
    #: The per-cell timing state the sweep recorded, summarised: how many of
    #: this replicate's cells were timed at a clock the roof would accept, and
    #: how many said nothing about their clock at all.
    timing_state: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Measured, readable, and non-empty. `REPORT_CODES` decides the first.

        THE SECOND CALL SITE OF THE `returncode == 0` BUG. `run_replicate`
        reads the same rule, and this property re-decided it independently, so
        the arm was discarded twice over and fixing either place alone left the
        other one discarding it.
        """
        return (self.returncode in REPORT_CODES and not self.error
                and bool(self.cells))


COMPILE_RE = re.compile(r"BM=(\d+):(\d+)")


def parse_compiles(doc: dict) -> dict[int, int]:
    """Fresh Triton artefacts per BLOCK_M, from the sweep's own gate 0 line.

    Returns {} when the line is not there or does not parse. {} is REFUSAL, not
    zero: the V5 gate reads an empty dict as UNKNOWN and never as "nothing
    compiled", because those two states have opposite meanings for cache freshness.
    """
    for gate in doc.get("gates", []):
        if gate.get("number") == 0:
            return {int(bm): int(n) for bm, n in COMPILE_RE.findall(gate.get("measured", ""))}
    return {}


def load_replicate(rep: Replicate) -> Replicate:
    """Read one replicate's report.json into ladder cells.

    THE INSTRUMENT IS READ OFF THE REPORT, not assumed from this process. A
    floor pooled over replicates timed by two different loops is a measurement
    of the difference between the loops, and this repo has just retired one
    instrument in favour of another: the committed arms part (b) reads were
    timed by the retired `time_call` and carry no instrument field at all,
    while anything a pod produces from now on carries
    `moe.bench.timing.TIMING_BASIS`. V6 scores the agreement; this only records
    what each replicate said.
    """
    if not rep.report.exists():
        rep.error = f"no report.json at {rep.report}"
        return rep
    doc = json.loads(rep.report.read_text())
    fixed = doc["fixed"]
    rep.compiles = parse_compiles(doc)
    rep.instrument = doc.get("instrument") or ""
    prov = doc.get("provenance") or {}
    rep.timing_state = {
        "instrument": rep.instrument,
        "warmup_ms": prov.get("warmup_ms"),
        "iters": prov.get("iters"),
        "target_ms": prov.get("target_ms"),
        "cells_excluded_for_clock_level": doc.get(
            "cells_excluded_for_clock_level"),
        "timing_spread_median": doc.get("timing_spread_median"),
    }
    for block_m, fit in doc["ladder"].items():
        rep.cells.append(LadderCell(
            model=doc["model"], dtype=doc["dtype"],
            group_m=fixed["GROUP_SIZE_M"], block_n=fixed["BLOCK_SIZE_N"],
            block_k=fixed["BLOCK_SIZE_K"], num_warps=fixed["num_warps"],
            num_stages=fixed["num_stages"], block_m=int(block_m),
            values={f: fit.get(f) for f in ALPHA_FIELDS},
            treads=fit.get("memory_points", 0), sm_count=doc.get("sm_count", 0),
            source=f"{rep.run_id}"))
    return rep


# --------------------------------------------------------------------------
# pooling replicates into a floor
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CellSpread:
    """One (arm, BLOCK_M) cell's between-replicate spread in one field."""

    arm: str
    block_m: int
    field: str
    values: tuple[float, ...]

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.values)

    @property
    def sd(self) -> float:
        return statistics.stdev(self.values) if self.n >= 2 else 0.0

    @property
    def cv(self) -> float | None:
        m = self.mean
        return self.sd / m if m else None

    @property
    def degenerate(self) -> bool:
        """All replicates bit-identical: a collision signature, not a floor."""
        return self.n >= 2 and len({round(v, 12) for v in self.values}) == 1


@dataclass(frozen=True)
class PooledFloor:
    """The floor for one alpha field, pooled or refused."""

    field: str
    spreads: tuple[CellSpread, ...]
    pooled_sd: float | None
    df: int
    pooled: bool
    reason: str

    @property
    def upper95(self) -> float | None:
        if self.pooled_sd is None or self.df < 1:
            return None
        return sd_upper_bound(self.pooled_sd, self.df)


def pool(spreads: list[CellSpread], field_name: str) -> PooledFloor:
    """Pool within-cell variance across cells, or refuse and publish the widest.

    Pooling is only legitimate when the cells share a spread. The homogeneity
    test is max(sd)/min(sd) <= 3, and a FAIL does not abort: it falls back to the
    WIDEST cell, which is the conservative floor, and says so. Averaging a wild
    cell down into a tight one would publish a floor that declares its own
    outlier resolvable.
    """
    usable = [s for s in spreads if s.n >= 2]
    if not usable:
        return PooledFloor(field_name, tuple(spreads), None, 0, False,
                           "no cell had two or more replicates")
    sds = [s.sd for s in usable]
    if min(sds) <= 0:
        widest = max(usable, key=lambda s: s.sd)
        return PooledFloor(
            field_name, tuple(spreads), widest.sd if widest.sd > 0 else None,
            widest.n - 1, False,
            f"at least one cell had zero spread, which is a collision signature "
            f"rather than a floor; the widest cell ({widest.arm} "
            f"BM={widest.block_m}) is published instead")
    ratio = max(sds) / min(sds)
    if ratio > HOMOGENEITY_RATIO:
        widest = max(usable, key=lambda s: s.sd)
        return PooledFloor(
            field_name, tuple(spreads), widest.sd, widest.n - 1, False,
            f"spreads disagree by {ratio:.2f}x (> {HOMOGENEITY_RATIO:g}), so the "
            f"cells were NOT pooled; the widest cell "
            f"({widest.arm} BM={widest.block_m}) is the published floor")
    ss = sum((s.n - 1) * s.sd ** 2 for s in usable)
    df = sum(s.n - 1 for s in usable)
    return PooledFloor(field_name, tuple(spreads), math.sqrt(ss / df), df, True,
                       f"pooled over {len(usable)} cells, spreads within "
                       f"{ratio:.2f}x of one another")


def spreads_for(replicates: list[Replicate], field_name: str) -> list[CellSpread]:
    """Group finished replicates into (arm, BLOCK_M) cells for one field."""
    buckets: dict[tuple[str, int], list[float]] = {}
    for rep in replicates:
        if not rep.ok:
            continue
        for cell in rep.cells:
            value = cell.values.get(field_name)
            if value is None:
                continue
            buckets.setdefault((rep.arm, cell.block_m), []).append(value)
    return [CellSpread(arm, bm, field_name, tuple(vals))
            for (arm, bm), vals in sorted(buckets.items())]


def two_sample_delta(spreads: list[CellSpread], first_arm: str, second_arm: str,
                     block_m: int) -> tuple[float, float, int] | None:
    """(delta, pooled sd, n per condition) for one BLOCK_M across two arms."""
    a = next((s for s in spreads if s.arm == first_arm and s.block_m == block_m), None)
    b = next((s for s in spreads if s.arm == second_arm and s.block_m == block_m), None)
    if a is None or b is None or a.n < 2 or b.n < 2:
        return None
    df = a.n + b.n - 2
    sd = math.sqrt(((a.n - 1) * a.sd ** 2 + (b.n - 1) * b.sd ** 2) / df)
    return b.mean - a.mean, sd, min(a.n, b.n)


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Gate:
    """One pre-registered prediction, its expected verdict, and what happened.

    `expected` is filled in BEFORE the run and printed beside the outcome, so a
    FAIL that was predicted reads as a result and a PASS that was not predicted
    reads as a surprise worth chasing. `passed=None` prints UNKNOWN and never
    counts as a pass, and `moe.bench.exit_codes.classify` scores UNKNOWN against
    the gate for the same reason: a check that examined nothing also reports
    zero failures.

    `kind` is VALIDITY or CLAIM in `exit_codes`'s own vocabulary, because the
    two have opposite consequences -- a VALIDITY non-PASS means nothing on the
    page may be quoted (INVALID), a CLAIM non-PASS is the world disagreeing with
    a pre-registered expectation (CLAIM_FAIL, which is a result and never a
    retry).
    """

    kind: str
    name: str
    prediction: str
    rule: str
    expected: str
    passed: bool | None
    observed: str
    invalidates: str = ""

    @property
    def verdict(self) -> str:
        return {True: exit_codes.PASS, False: exit_codes.FAIL,
                None: exit_codes.UNKNOWN}[self.passed]

    @property
    def token(self) -> str:
        """The one-token name this gate answers to on its `RESULT:` line.

        `result_line` refuses a name with whitespace, and these names are
        sentences ("C1 floor size"), so the spaces become underscores here
        rather than the gate silently disappearing from the driver's summary.
        """
        return re.sub(r"\s+", "_", self.name.strip())

    def scored(self) -> tuple[str, str, str]:
        return (self.kind, self.token, self.verdict)

    def result_line(self) -> str:
        """The ONE line a driver may grep for this gate.

        Nothing else this file prints starts with `RESULT: `. The summary that
        greps `floor|sigma` matched a REFUSED log eighteen times and printed the
        imported proxy and a pre-registered expectation as measured output; this
        line is what that grep should have been reading. It is not what the
        session driver reads yet, so this line OFFERS the fix rather than being
        it: `arm_gate_regex` still selects free text for this arm.
        """
        detail = (f"[{self.kind}] {self.prediction} | expected {self.expected} "
                  f"| gate {self.rule} | saw {self.observed}")
        return exit_codes.result_line(*self.scored(), " ".join(detail.split()))

    def render(self) -> str:
        surprise = ""
        if self.passed is not None:
            got = "PASS" if self.passed else "FAIL"
            surprise = "" if got == self.expected else f"  <-- expected {self.expected}"
        out = [self.result_line(),
               f"[{self.verdict}] {self.kind:8s} {self.name}  "
               f"{self.prediction}{surprise}",
               f"         gate: {self.rule}",
               f"         saw:  {self.observed}"]
        if self.passed is False and self.invalidates:
            out.append(f"         a FAIL here invalidates: {self.invalidates}")
        return "\n".join(out)


def render_gates(gates: list[Gate]) -> str:
    lines = [g.render() for g in gates]
    lines.append("")
    lines.append(f"{sum(1 for g in gates if g.passed is True)} PASS, "
                 f"{sum(1 for g in gates if g.passed is False)} FAIL, "
                 f"{sum(1 for g in gates if g.passed is None)} UNKNOWN")
    return "\n".join(lines)


def rehearsal_exit(rc: int, gates: Sequence[Gate]) -> tuple[int, str]:
    """`(the code a REHEARSAL may exit with, the sentence that says why)`.

    A rehearsal runs the sweep's `--self-test`, so its cells are GENERATED from
    a model and nothing on the page was measured. It must never reach a
    quotable code, and until 2026-09-03 the only thing stopping it was V6
    noticing that the synthetic cells carry the synthetic instrument stamp.
    That is one wall, standing in a gate whose job is something else, and it
    happens to be load-bearing.

    WHY IT NEEDED A SECOND ONE, AND WHY THIS IS NOT BELT AND BRACES. Reading a
    CLAIM_FAIL child's report is exactly what the fix above this made the parent
    start doing, and it is what turns a rehearsal from "V2 FAIL, nothing pooled"
    into a run whose gates now genuinely score. From that commit a rehearsal is
    ONE stamp away from printing `exit 0 DONE` under a floor that is whatever
    `--rehearse-noise` was set to. So the code is decided here, from the fact of
    the rehearsal, and V6 becomes the explanation rather than the mechanism.

    IT ALSO ANSWERS A QUESTION THE OPERATOR ASKS ON A METERED POD. The last line
    a rehearsal prints is `exit 3 INVALID`, the identical line a real arm prints
    when the instrument broke while in use, and the banner that explained the
    difference scrolled past several hundred lines earlier. The reason travels
    with the code now.
    """
    carried = [g.name for g in gates
               if g.kind == exit_codes.VALIDITY and g.passed is not True]
    if rc == exit_codes.INVALID:
        return rc, ("REHEARSAL: INVALID by construction, and that is the gates "
                    "working. Nothing here was measured; the floor above is "
                    "whatever --rehearse-noise was set to. Carried by: "
                    + ", ".join(carried))
    return exit_codes.INVALID, (
        f"REHEARSAL: the gates classified this run {exit_codes.describe(rc)}, "
        "which a rehearsal may not exit with, so it is forced to INVALID. "
        "Every VALIDITY gate PASSED over cells that were GENERATED, which is "
        "not a pass: it means the wall that keeps synthetic cells out of a "
        "quotable page (V6, the instrument stamp) stopped working. Read that "
        "as a defect in the rehearsal plumbing, not as a floor.")


def gates_permit_publishing(gates: Sequence[Gate], *, floor_withheld: bool,
                            cache_mode: str, floor_from: str) -> tuple[bool, str]:
    """`(may this MEASURED run write the TRACKED floor, and if not, why not)`.

    THE GATES REFUSED THE RUN AND `--publish` WROTE IT ANYWAY, because the only
    thing that branch asked was whether the run was a rehearsal. The exit code
    told the truth -- a run with a failed VALIDITY gate exits `INVALID`,
    "nothing quotable" -- and the tracked floor was written on the way there.
    The session driver runs this arm WITH `--publish`, so an arm that
    failed V6 (two different timing instruments pooled into one spread) or V1
    (two replicates that resumed one directory) would have left an unquotable
    number in the curated directory under a filename every other script imports
    by, with nothing in the file to say the gates had refused it. The log said
    INVALID; git got a floor.

    ONLY VALIDITY BLOCKS, AND THAT IS THE WHOLE DISTINCTION THE TABLE MAKES. A
    CLAIM gate that failed is a RESULT: the spread was measured soundly and the
    world disagreed with a pre-registered prediction about it, and the spread is
    what this file publishes. A VALIDITY gate that failed says the spread itself
    means nothing. Blocking on a CLAIM would throw away the artefact the card
    was rented for; not blocking on a VALIDITY publishes a number no one may
    quote. UNKNOWN blocks with FAIL: a validity gate that could not decide has
    not established that the run is quotable, which is exactly how
    `exit_codes.classify` scores it.

    AND THE SECOND WAY IN IS THE ONE THAT WRITES NOTHING AT ALL. `--warm-cache` with
    `--floor-from fresh` measures cells whose floor is deliberately withheld from
    the document, so `build_document` is handed `floors=None` and the run's
    `replicate_floor` is null. It passed every gate, so a gates-only wall waves
    it through, and it OVERWRITES a measured floor from an earlier run with a
    null. The page already printed "NOT publishable as THE floor" and then
    published anyway. A withheld run's document is field for field the one
    `--control-only --publish` writes, so refusing it costs nothing and the
    operator is told which mode to rerun in.

    NEITHER RULE GOVERNS `--control-only --publish`. That path has no cells and
    no gates, and writing `replicate_floor: null` is the whole of what it claims;
    part (b) is arithmetic over two committed arms and has no floor to
    invalidate. A rule applied there would refuse the one publish path that is
    always legitimate.
    """
    failed = [f"{g.name} ({'UNKNOWN' if g.passed is None else 'FAIL'})"
              for g in gates
              if g.kind == exit_codes.VALIDITY and g.passed is not True]
    if failed:
        return False, (
            "VALIDITY gates that did not pass: " + ", ".join(failed)
            + ". A VALIDITY gate that did not pass says this run's spread means "
              "nothing. A CLAIM gate failing would NOT have stopped this: that "
              "is a result about the floor, not a reason to doubt it.")
    if floor_withheld:
        return False, (
            f"this run measured the {cache_mode} cache and --floor-from is "
            f"{floor_from}, so its floor is withheld from the document and the "
            f"file it would write carries replicate_floor: null. Publishing it "
            f"would replace a measured floor with nothing. Rerun with "
            f"--floor-from {cache_mode}, or publish the {floor_from} run.")
    return True, ""


def validity_gates(replicates: list[Replicate], expected_n: int, arms: list[Arm],
                   cache_mode: str, floors: dict[str, PooledFloor],
                   *, single_model_ok: bool = False) -> list[Gate]:
    """V-gates. A FAIL means no number from part (a) may be quoted."""
    gates: list[Gate] = []
    done = [r for r in replicates if r.ok]

    gates.append(Gate(
        exit_codes.VALIDITY, "V1 distinct runs",
        "every replicate wrote its own directory under its own run id",
        f"{len(replicates)} distinct run ids and {len(replicates)} distinct out dirs",
        "PASS",
        None if not replicates else (
            len({r.run_id for r in replicates}) == len(replicates)
            and len({str(r.out_dir) for r in replicates}) == len(replicates)),
        "nothing was launched" if not replicates else
        f"{len({r.run_id for r in replicates})} ids and "
        f"{len({str(r.out_dir) for r in replicates})} dirs over {len(replicates)} replicates",
        "the whole floor: colliding run ids make the sweep resume into itself and "
        "report one run's timings N times"))

    want = expected_n * len(arms)
    gates.append(Gate(
        exit_codes.VALIDITY, "V2 non-vacuity",
        "the replicates that were planned actually ran and produced fits",
        f"{want} replicates complete and at least 2 cells with all "
        f"{expected_n} values present",
        "PASS",
        None if not replicates else (
            len(done) == want
            and len([s for s in floors[PRIMARY_FIELD].spreads
                     if s.n == expected_n]) >= 2),
        f"{len(done)} of {want} replicates ok ({code_census(replicates)}); "
        f"{len([s for s in floors[PRIMARY_FIELD].spreads if s.n == expected_n])} "
        f"full cells of {len(floors[PRIMARY_FIELD].spreads)} seen"
        if replicates else "nothing was launched",
        "everything below: a check that examined nothing also reports zero "
        "failures"))

    degenerate = [s for s in floors[PRIMARY_FIELD].spreads if s.degenerate]
    gates.append(Gate(
        exit_codes.VALIDITY, "V3 not a collision",
        "no cell returned bit-identical alpha in every replicate",
        "zero cells with exactly one distinct value across replicates",
        "PASS",
        None if not floors[PRIMARY_FIELD].spreads else len(degenerate) == 0,
        "no cell was measured" if not floors[PRIMARY_FIELD].spreads else
        (f"{len(degenerate)} degenerate cells"
         + ("" if not degenerate
            else f"; first {degenerate[0].arm} BM={degenerate[0].block_m}")),
        "the floor: identical alphas mean the replicates resumed into one "
        "directory, and the sd is 0.0000 for a reason that is not physics"))

    if cache_mode == "fresh":
        counts = [min(r.compiles.values()) for r in done if r.compiles]
        gates.append(Gate(
            exit_codes.VALIDITY, "V4 fresh cache",
            "every replicate compiled its own Triton artefacts at every setting",
            "minimum fresh-artefact count over replicates and settings >= 1",
            "PASS",
            None if not counts else min(counts) >= 1,
            "no replicate reported a compile count" if not counts
            else f"minimum {min(counts)} artefacts over {len(counts)} replicates",
            "the unit: a warm cache makes this a narrower floor than the "
            "comparisons it is meant to score"))
    else:
        # Replicates are launched INTERLEAVED, so "everything after the first"
        # is not a tail slice of the list and never was a per-arm one either:
        # it is every replicate whose INDEX is above 1, which is one exclusion
        # per arm however the arms are ordered within a replicate.
        later = [min(r.compiles.values()) for r in done if r.index > 1 and r.compiles]
        gates.append(Gate(
            exit_codes.VALIDITY, "V4 warm cache",
            "replicates after the first reused the shared Triton cache",
            "minimum fresh-artefact count over replicates 2..N == 0",
            "PASS",
            None if not later else min(later) == 0,
            "no later replicate reported a compile count" if not later
            else f"minimum {min(later)} artefacts over {len(later)} later replicates",
            "the label: a 'warm' floor that recompiled every time is the fresh "
            "floor under the wrong name"))

    ratios = {}
    for name, floor in floors.items():
        sds = [s.sd for s in floor.spreads if s.n >= 2 and s.sd > 0]
        if len(sds) >= 2:
            ratios[name] = max(sds) / min(sds)
    gates.append(Gate(
        exit_codes.VALIDITY, "V5 homogeneity",
        "the cells share a spread, so pooling them into one floor is legitimate",
        f"max cell sd / min cell sd <= {HOMOGENEITY_RATIO:g} in {PRIMARY_FIELD}",
        "PASS",
        None if PRIMARY_FIELD not in ratios
        else ratios[PRIMARY_FIELD] <= HOMOGENEITY_RATIO,
        "fewer than two cells had a positive spread" if PRIMARY_FIELD not in ratios
        else f"{ratios[PRIMARY_FIELD]:.2f}x spread across cells",
        "nothing -- a FAIL falls back to the WIDEST cell, which is the "
        "conservative floor, and the published number says so"))

    # V6. ONE INSTRUMENT, OR NO FLOOR. Two replicates timed by two loops
    # measure the difference between the loops, and this repo retired one
    # instrument for another on 2026-09-02: the committed arms part (b) reads
    # carry NO instrument field, everything a pod produces from now on carries
    # `moe.bench.timing.TIMING_BASIS`. An empty string is a DIFFERENT state
    # from a named-but-other instrument and both fail here, because a floor
    # pooled across either is not a floor.
    stamps = {r.instrument for r in done}
    expected_basis = timing_basis()
    named = expected_basis or "the instrument could not be named on this host"
    gates.append(Gate(
        exit_codes.VALIDITY, "V6 one instrument",
        "every replicate was timed by the instrument this repo publishes with",
        f"every report stamps instrument == {named!r}",
        "PASS",
        None if not stamps or expected_basis is None
        else stamps == {expected_basis},
        "nothing was launched" if not stamps
        else ("the instrument could not be named on this host, so agreement "
              "cannot be checked" if expected_basis is None
              else "instruments seen: "
                   + ", ".join(sorted(repr(s) for s in stamps))),
        "the floor and every contrast built on it: a spread pooled over two "
        "timing loops measures the loops, and an unstamped report is a report "
        "from before the instrument had a name rather than one that agrees"))

    # V7. THE FLOOR'S SCOPE IS PART OF THE FLOOR. A floor measured on the one
    # model where the swizzle effect is 0.3855 does not license a surface
    # published across four models, and until 2026-09-02 that was the whole
    # design. A single-model floor is still available and still refused
    # silently: it has to be asked for in argv, and the scope block says so.
    models = sorted({a.model for a in arms})
    gates.append(Gate(
        exit_codes.VALIDITY, "V7 scope",
        "the floor covers more than the one model where the effect is largest",
        ">= 2 models among the arms, or --single-model-floor in argv",
        "PASS",
        len(models) >= 2 or single_model_ok,
        f"models measured: {', '.join(models) or 'none'}"
        + ("; --single-model-floor was given" if single_model_ok else ""),
        "the generality of everything scored against this floor: the qwen2 "
        "swizzle effect is 0.0226 where mixtral's is 0.3855, and a floor that "
        "never saw qwen2 cannot say which of the two the surface is about"))
    return gates


def swizzle_gate_name(model: str) -> str:
    """`C3 swizzle mixtral-8x7b`: one C3 per model, named so it stays distinct.

    The old C3 took `arms[0]` against `arms[-1]`, which was the two ends of the
    swizzle while the arms were one model and became mixtral-G=1 against
    qwen2-G=16 the moment a second model was added -- a contrast across two
    levers at once, scored as though it were one.
    """
    return f"C3 swizzle {model}"


def claim_gates(floors: dict[str, PooledFloor], spreads: list[CellSpread],
                control: dict[str, PairedDifference],
                cards: dict[str, PairedDifference],
                arms: list[Arm]) -> list[Gate]:
    """C-gates. A FAIL here is a result, not a broken run."""
    gates: list[Gate] = []
    primary = floors.get(PRIMARY_FIELD)
    sd = primary.pooled_sd if primary else None

    gates.append(Gate(
        exit_codes.CLAIM, "C1 floor size",
        f"the measured floor is no wider than the s3/s4 proxy implied "
        f"({prior_sd():.4f})",
        f"pooled between-replicate sd of {PRIMARY_FIELD} <= {prior_sd():.4f}",
        "PASS",
        None if sd is None else sd <= prior_sd(),
        "part (a) did not run" if sd is None
        else f"pooled sd {sd:.4f} on {primary.df} df, upper 95% "
             f"{primary.upper95:.4f}",
        "every N chosen against the prior: a wider floor means the whole study "
        "is scored against a looser instrument than it assumed"))

    delta_card = cards[PRIMARY_FIELD]
    mde_card = mde_at_measured_floor(sd, 1, external=True)
    gates.append(Gate(
        exit_codes.CLAIM, "C2 cross-card under floor",
        "the published cross-card difference is smaller than one run per card "
        "could resolve",
        "|paired mean| < the known-sigma MDE at the measured floor, n=1 per card",
        "PASS",
        None if mde_card is None or delta_card.mean is None
        else abs(delta_card.mean) < mde_card,
        ("part (a) did not run" if sd is None
         else "the measured floor is not positive, so no difference can be "
              "priced against it; V3 says whether that is a collision")
        if mde_card is None or delta_card.mean is None
        else (f"|{delta_card.mean:+.4f}| against MDE "
              f"{mde_card:.4f} at the design the study ran, "
              f"one run per card"),
        "nothing -- a PASS here is the finding: the cross-card result is not a "
        "null, it is an unresolvable measurement"))

    # ONE C3 PER MODEL, each scored against ITS OWN registered effect size and
    # its OWN replicates. mixtral's is expected to clear the floor by a factor
    # of nine; qwen2's is 0.0226 against an MDE near 0.041 and is registered as
    # an EXPECTED FAIL, which is the whole reason the qwen2 arm exists. A
    # single pooled C3 would have reported the mixtral answer for both.
    for model, group in swizzle_pairs(arms).items():
        registered = next((e for e in EFFECTS
                           if e.model == model and e.name == "swizzle swing"),
                          None)
        expected = "PASS" if registered is None or registered.size >= 0.10 \
            else "FAIL"
        if len(group) < 2 or group[0].group_m == group[-1].group_m:
            gates.append(Gate(
                exit_codes.CLAIM, swizzle_gate_name(model),
                "the GROUP_SIZE_M swing on this model is bigger than the floor",
                "|delta| >= the two-sample MDE at every measured BLOCK_M",
                expected, None,
                f"{model} was measured at one GROUP_SIZE_M only, so there is "
                "no swizzle contrast to score",
                "this model's row of the alpha surface"))
            continue
        lo, hi = group[0], group[-1]
        resolvable = []
        # BLOCK_M values that were measured and whose two arms agreed to the
        # bit. They are NOT unmeasured and must not be reported as such: they
        # are V3's collision, and the observed line below says so instead of
        # letting them vanish out of the count.
        collided = []
        for block_m in sorted({s.block_m for s in spreads}):
            got = two_sample_delta(spreads, lo.name, hi.name, block_m)
            if got:
                delta, pooled_sd, n = got
                limit = mde_at_measured_floor(pooled_sd, n, external=False)
                if limit is None:
                    collided.append(block_m)
                else:
                    resolvable.append((block_m, delta, limit))
        collision_note = ("" if not collided else
                          "; BM " + ",".join(str(b) for b in collided)
                          + " had a zero spread and could not be priced "
                            "(see V3)")
        gates.append(Gate(
            exit_codes.CLAIM, swizzle_gate_name(model),
            f"the GROUP_SIZE_M {lo.group_m} -> {hi.group_m} swing on {model} "
            f"is bigger than the floor at every BLOCK_M"
            + ("" if registered is None
               else f" (registered at {registered.size:.4f})"),
            "|delta| >= the two-sample MDE at every measured BLOCK_M",
            expected,
            all(abs(d) >= m for _, d, m in resolvable) if resolvable else None,
            (("nothing was measured for this model" if not collided else
              "every measured BLOCK_M collided") + collision_note)
            if not resolvable else
            "; ".join(f"BM={bm} {delta:+.4f} vs MDE {m:.4f}"
                      for bm, delta, m in resolvable) + collision_note,
            "this model's row of the alpha surface: a swizzle swing inside the "
            "floor is a surface feature that is noise"))

    ctrl = control[PRIMARY_FIELD]
    card = cards[PRIMARY_FIELD]
    separable = None
    if ctrl.mean is not None and card.mean is not None and ctrl.mde is not None:
        separable = abs(card.mean) - abs(ctrl.mean) >= ctrl.mde
    gates.append(Gate(
        exit_codes.CLAIM, "C4 card beats stages control",
        "changing the card moves alpha more than changing num_stages on one card",
        "|cross-card mean| - |stages-control mean| >= the control's own paired MDE",
        "FAIL",
        separable,
        "one of the two arms is missing" if separable is None else
        (f"|{card.mean:+.4f}| - |{ctrl.mean:+.4f}| = "
         f"{abs(card.mean) - abs(ctrl.mean):+.4f} against the control's MDE "
         f"{ctrl.mde:.4f} on n={ctrl.n}"),
        "the L2 reading of the cross-card arm: a pipelining change on ONE card "
        "moves alpha as much as a 1.5x change in L2 capacity, so the cross-card "
        "difference cannot be attributed to L2"))

    signs = {}
    for name, diff in cards.items():
        if diff.mean is not None:
            signs[name] = math.copysign(1.0, diff.mean)
    consistent = None
    if len(signs) == len(ALPHA_FIELDS):
        consistent = len(set(signs.values())) == 1
    ctrl_signs = {n: math.copysign(1.0, d.mean)
                  for n, d in control.items() if d.mean is not None}
    gates.append(Gate(
        exit_codes.CLAIM, "C5 sign consistency",
        "the cross-card difference points the same way in every alpha estimator",
        "sign(alpha) == sign(alpha_corrected) == sign(alpha_upper)",
        "FAIL",
        consistent,
        "not every field was differenced" if consistent is None else
        ("cross-card " + ", ".join(f"{n} {cards[n].mean:+.4f}" for n in ALPHA_FIELDS)
         + "  |  control " + ", ".join(
             f"{n} {control[n].mean:+.4f}" for n in ALPHA_FIELDS
             if control[n].mean is not None)
         + f"  (control signs agree: {len(set(ctrl_signs.values())) == 1})"),
        "the cross-card result in EITHER direction: an effect whose sign depends "
        "on which of three anchorings of the same fit you read is not an effect"))
    return gates


# --------------------------------------------------------------------------
# writing where git will take it
# --------------------------------------------------------------------------

def git_accepts(path: Path) -> bool | None:
    """Would git track a file at this path? None when git cannot say.

    `results/*` is ignored with only `!results/published/` excepted, so a summary
    written one directory over is a summary that vanishes on commit. The rule is
    checked here rather than trusted, and an inconclusive answer (no git, no
    repo) is None and is treated as a refusal by the caller.
    """
    try:
        done = subprocess.run(["git", "check-ignore", "-q", str(path)],
                              cwd=ROOT, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode == 0:
        return False
    if done.returncode == 1:
        return True
    return None


#: How many dirty paths a published floor lists before it stops listing them.
#: A boolean is not enough and the whole `git status` is too much: a reader of
#: the committed file has to be able to see WHETHER THE DIRT MATTERED.
DIRTY_PATHS_LISTED = 20


def git_state() -> dict:
    """Commit, dirtiness, and WHICH paths were dirty, for the published floor.

    `dirty` USED TO BE A BARE BOOLEAN AND THE COMMITTED FLOOR CARRIED `true`.
    That is a fact a reader can do nothing with. "The tree was dirty" reads the
    same whether the uncommitted change was a stray notebook or the very
    arithmetic that produced the number, and the audit could only record it as
    an unexplained flag beside a published quantity.

    IT CANNOT BE FALSE ON THE COMMIT THAT PUBLISHES, AND THAT IS ARITHMETIC,
    NOT SLOPPINESS. The file has to be written before it can be committed, so
    the tree that produces it always contains at least the change being
    committed, and a document naming the commit that contains itself does not
    exist. So the honest artefact names the parent commit, says what was
    outstanding against it, and leaves the reader to judge. The stronger
    guarantee is not this block at all: it is
    `tests/test_replicate_noise_floor.py::
    test_the_committed_floor_reproduces_from_the_committed_code`, which
    recomputes every field of the committed file from the tracked arms and the
    tracked code and fails on a byte of drift.
    """
    out = {"commit": "", "dirty": None, "dirty_paths": None}
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=30)
        if rev.returncode == 0:
            out["commit"] = rev.stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        if status.returncode == 0:
            lines = [ln for ln in status.stdout.splitlines() if ln.strip()]
            out["dirty"] = bool(lines)
            paths = [ln[3:].strip() for ln in lines[:DIRTY_PATHS_LISTED]]
            if len(lines) > DIRTY_PATHS_LISTED:
                paths.append(f"... and {len(lines) - DIRTY_PATHS_LISTED} more")
            out["dirty_paths"] = paths
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def paired_payload(diff: PairedDifference) -> dict:
    return {"first": diff.first, "second": diff.second,
            "sign": f"delta = alpha({diff.second}) - alpha({diff.first})",
            "n_cells": diff.n, "mean": diff.mean, "sd": diff.sd,
            "mde": diff.mde, "resolved": diff.resolved,
            "varied": list(diff.varied), "same_machine": diff.same_machine}


def by_model_spread(diff: PairedDifference) -> dict:
    """The proxy's paired deltas split by MODEL, so its heteroscedasticity ships.

    "Inside 2x the pooled 0.0323" was the yardstick every effect in the study
    was read against, and the pool is not one population: mixtral's four deltas
    have sd 0.0486 and qwen2's seven have 0.0186. A per-cell comparison at
    mixtral G>=8 against the pooled number is against a spread half again too
    narrow. Computed here from the same pairs rather than restated, so the
    published block cannot drift from the arms it came from.
    """
    per: dict[str, list[float]] = {}
    for key, a, b in diff.pairs:
        per.setdefault(str(key[0]), []).append(b - a)
    out = {}
    for model, deltas in sorted(per.items()):
        out[model] = {
            "n": len(deltas),
            "mean": statistics.fmean(deltas),
            "sd": statistics.stdev(deltas) if len(deltas) >= 2 else None,
            "max_abs": max(abs(d) for d in deltas),
        }
    sds = [v["sd"] for v in out.values() if v["sd"]]
    return {"per_model": out,
            "ratio": (max(sds) / min(sds)) if len(sds) >= 2 else None,
            "homogeneity_ratio": HOMOGENEITY_RATIO}


def scope_block(arms: list[Arm], *, n_replicates: int, cache_mode: str,
                gpu_name: str, order: str, single_model_ok: bool) -> dict:
    """What a floor from this run does and does not cover, in the file itself.

    A floor whose scope lives only in a docstring gets quoted without it, and
    that is what happened: the proxy was labelled "upper bound only" with no
    statement of what it bounded, and was then cited for cross-card, cross-pod
    and other-model comparisons it says nothing about.
    """
    models = sorted({a.model for a in arms})
    return {
        "session": "ONE card, ONE session. Nothing here bounds cross-pod, "
                   "cross-day or cross-card noise; replicates that span "
                   "sessions would be a different measurement and this is not "
                   "it",
        "gpu_name": gpu_name,
        "models": models,
        "single_model_floor_allowed": single_model_ok,
        "arms": [{"name": a.name, "model": a.model, "group_m": a.group_m,
                  "block_n": a.block_n, "num_stages": a.num_stages,
                  "dtype": a.dtype, "seed": a.seed,
                  "warmup_ms": a.warmup_ms, "trials": a.trials,
                  "l2_flush": a.l2_flush} for a in arms],
        "block_n": sorted({a.block_n for a in arms}),
        "num_stages": sorted({a.num_stages for a in arms}),
        "dtype": sorted({a.dtype for a in arms}),
        "seed_held_fixed": sorted({a.seed for a in arms}),
        "n_replicates": n_replicates,
        "cache_mode": cache_mode,
        "order": order,
        "instrument": timing_basis(),
        "excludes": [
            "data-generation variance: the seed is held fixed across "
            "replicates, so this floor may not score a comparison that "
            "re-rolled its inputs",
            "any BLOCK_SIZE_N, num_stages, dtype or model not listed above",
            "anything timed by an instrument other than the one named above",
        ],
    }


def build_document(control: dict[str, PairedDifference],
                   cards: dict[str, PairedDifference],
                   floors: dict[str, PooledFloor] | None,
                   prov: PV.Provenance,
                   *, n_replicates: int = 0, cache_mode: str = "",
                   gpu_name: str = "", provenance: str = "",
                   synthetic: bool = False, scope: dict | None = None) -> dict:
    """The published JSON. `replicate_floor` is null until a card produces one.

    `prov` IS POSITIONAL AND REQUIRED AS OF 2026-09-02. It was a keyword
    defaulting to None, and two of the three call sites did not pass it: the
    `--control-only --publish` path and the no-GPU/`--dry-run --publish` path
    each wrote the TRACKED file `results/published/NOISE_FLOOR.json` with no
    provenance block at all, so the committed floor named no machine, no
    instrument, no ridge source and no bandwidth source. The only attribution
    left was the home-grown `git_state()`, which is precisely the "re-invented a
    subset" shape `moe.bench.provenance` exists to end. A default is what three
    call sites forget; a required argument is what none of them can. There is a
    block for a page that measured nothing, `unmeasured_provenance()`, so the
    requirement is never a reason to invent one.
    """
    doc = {
        "schema": SCHEMA,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git_state(),
        "sign": "every delta is alpha(second arm) - alpha(first arm)",
        "primary_field": PRIMARY_FIELD,
        "effects_registered": [{"name": e.name, "size": e.size,
                                "source": e.source, "model": e.model}
                               for e in EFFECTS],
        "prior_sd": prior_sd(),
        "prior_sd_source": PRIOR_SD_SOURCE,
        "prior_sd_by_model": by_model_spread(control[PRIMARY_FIELD]),
        "scope": scope,
        "stages_control": {name: paired_payload(d) for name, d in control.items()},
        "stages_control_instrument":
            "the two committed arms were timed by the RETIRED per-iteration "
            "loop (block_m_crossing_sweep.time_call), not by "
            "moe.bench.timing.time_kernel: they carry no instrument field at "
            "all. Part (b) is arithmetic over those numbers and inherits their "
            "instrument",
        "cross_card": {name: paired_payload(d) for name, d in cards.items()},
        "replicate_floor": None,
    }
    doc = prov.stamp(doc)
    if floors:
        per_field = {}
        for name, floor in floors.items():
            if floor.pooled_sd is None:
                continue
            per_field[name] = {
                "sd": floor.pooled_sd, "df": floor.df, "upper95": floor.upper95,
                "pooled": floor.pooled, "reason": floor.reason,
                "cells": len([s for s in floor.spreads if s.n >= 2]),
                "per_cell": [{"arm": s.arm, "block_m": s.block_m, "n": s.n,
                              "mean": s.mean, "sd": s.sd, "values": list(s.values)}
                             for s in floor.spreads],
            }
        if per_field:
            doc["replicate_floor"] = {
                "n_replicates": n_replicates, "cache_mode": cache_mode,
                "gpu_name": gpu_name,
                # `provenance` HERE is the one-line prose `Floor.provenance`
                # has read since this file was written; the machine-readable
                # block from `moe.bench.provenance` is at the TOP level, where
                # the audit's publish gate looks for it. Two different things
                # under one word, kept because renaming this one would silently
                # give every existing reader a KeyError.
                "provenance": provenance,
                "instrument": timing_basis(),
                "scope": scope,
                "synthetic": synthetic, "per_field": per_field,
            }
    return doc


def published_floor_is_measured(path: Path) -> bool | None:
    """Does the tracked file ALREADY hold a floor that a card produced?

    Three answers and not two. `False` is "there is nothing there to lose":
    no file, or a file whose `replicate_floor` is null, or one holding a
    REHEARSAL floor, which is generated and may be replaced by anything.
    `True` is "a card measured this". `None` is CANNOT TELL: the file is there
    and does not parse, or carries a schema this module does not read, and the
    caller must refuse on it rather than treat unreadable as empty. Deciding
    "no floor" from a file we failed to open is how a measurement gets deleted
    by a script that thought it was writing into a blank.
    """
    if not path.exists():
        return False
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get("schema") not in SCHEMA_READABLE:
        return None
    block = doc.get("replicate_floor")
    if not block:
        return False
    return not block.get("synthetic")


def write_published(doc: dict, path: Path | None = None) -> str:
    """Write the floor where git will take it, or refuse and say why.

    TWO WALLS, AND THE SECOND ONE IS HERE BECAUSE THERE ARE THREE DOORS. The
    first is git: `results/*` is ignored with only `!results/published/`
    excepted, so a floor written anywhere else is a floor that disappears on
    commit.

    The second is the one `gates_permit_publishing` names in its own docstring,
    "it OVERWRITES a measured floor from an earlier run with a null", and that
    wall was built in the caller, at the MEASURED publish path, in the commit
    that found it. It therefore covered one door of three. The other two:
    `--control-only --publish`, and the `blocked` path, which is `--dry-run
    --publish` OR a real `--publish` run on which `detect_gpu()` reports
    anything missing. Both build their document with `floors=None`, both then
    called this function, and neither consulted a gate, because neither has any
    gates to consult. Proved off GPU on 2026-09-03 by seeding a measured
    `replicate_floor` into the tracked file and running `--dry-run --publish`:
    the page printed "NOT A RESULT. The replicate floor was not measured",
    returned REFUSED, and left `replicate_floor: null` in git. The session
    driver's real branch (`h200_gaps_session.sh:1750`) passes `--publish` bare,
    so one CUDA probe coming back empty on the pod both refused the arm and
    deleted the floor a previous pod had paid 120 minutes for.

    So the rule lives at the ONE place every door opens onto, and it is stated
    as a property of the write rather than of the caller: a document with no
    `replicate_floor` may not replace a file that has one. It costs the
    legitimate cases nothing, because a null-floor document is field for field
    what `--control-only --publish` regenerates and the file it would replace
    would be identical but for `written_utc`, `git` and `provenance`.

    IT REFUSES ON "CANNOT TELL" TOO. `published_floor_is_measured` returns None
    for a file that does not parse, and the refusal fires on anything that is
    not a definite False, so an unreadable tracked floor is a thing an operator
    is told about rather than a thing this function silently flattens.

    A `SystemExit` CARRYING A SENTENCE, like the git wall above it, and not a
    returned string: every caller of this prints what it returns, and a
    returned refusal at the measured path would leave the run exiting DONE over
    a floor that never landed. `main` maps a string exit to REFUSED, so the
    code says nothing was published and the log says why.
    """
    path = path or NOISE_FLOOR_JSON
    accepted = git_accepts(path)
    if accepted is not True:
        why = ("git ignores it" if accepted is False
               else "git could not be asked whether it ignores it")
        raise SystemExit(
            f"REFUSING to write {path}: {why}. The rule is `results/*` ignored "
            f"with only `!results/published/` excepted; a floor written anywhere "
            f"else is a floor that disappears on commit.")
    standing = published_floor_is_measured(path)
    if doc.get("replicate_floor") is None and standing is not False:
        held = ("holds a floor measured on a card" if standing
                else "cannot be read as a floor document of this schema")
        raise SystemExit(
            f"REFUSING to write {path}: this document carries "
            f"replicate_floor: null and the file already there {held}. "
            f"Publishing it would replace a measurement with nothing. Part (b) "
            f"is unchanged in both files; if the standing floor really is to go, "
            f"delete it in a commit that says so, then rerun this.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return f"wrote {path} (git check-ignore says this path is tracked)"


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

IMPORT_BANNER = f"""\
## How another script uses this

    import importlib.util, sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "replicate_noise_floor",
        Path("scripts/replicate_noise_floor.py").resolve())
    NF = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = NF
    spec.loader.exec_module(NF)

    NF.assert_resolvable(delta, "my cross-card claim")   # raises if too small
    floor = NF.noise_floor()                             # raises if unmeasured
    control = NF.stages_control()                        # needs no GPU

`assert_resolvable` raises `EffectBelowNoiseFloor` for anything the instrument
cannot see and `NoiseFloorUnmeasured` while part (a) has not run. Neither ever
returns a number nobody measured, so a caller that forgets to handle them stops
rather than silently deciding its effect is real.

AND FOR A CALLER THAT MUST PRINT A LIMIT EVERY SESSION, MEASURED OR NOT:

    sigma, basis, source = NF.sizing_sigma()   # basis is MEASURED or ASSUMED

The two consumers that still read `prior_sd` straight out of the JSON
(`scripts/alpha_surface.py:prior_sd`, `scripts/bn_decomposition.py:
published_prior_sd`) are reading the s3/s4 PROXY, and will still be reading it
after this arm publishes a measured floor into the same file. One line each.

The file is {NOISE_FLOOR_JSON.relative_to(ROOT)}."""


SIGN_BANNER = """\
THE SIGN, so it cannot be misread. Every difference below is

    delta = alpha(the arm named SECOND) - alpha(the arm named FIRST)

and every printed delta names both arms in that order. A positive delta means the
SECOND arm measured a HIGHER re-read fraction."""


def render_power_table(sd: float, label: str, cells: int = 4) -> str:
    """Detection limits, the sd's own confidence, and which one binds N.

    Both columns matter and they bind at different N. Power alone is satisfied at
    N=2 for the two big effects, which is why a power-only argument would have
    justified running the cheapest possible experiment and publishing a floor
    known only to a factor of six. The sd bound is what actually sets N=6.
    """
    rows = [f"Detection limits at sd = {sd:.4f} ({label}), two-sided "
            f"{TEST_LEVEL:.0%} at {TEST_POWER:.0%} power.",
            f"The last column is how well N pins the FLOOR itself, pooled over "
            f"{cells} cells.",
            "",
            # NOT WRITTEN AS `floor: <number>`. That was the shape of this
            # line until 2026-09-02, and the session driver's free-text summary
            # grep (`floor|sigma`) lifted it out of a REFUSED log and printed
            # the imported proxy as though this run had measured it. Reshaping
            # it stops it LOOKING like a result to a reader and to
            # `parse_result_lines`; it does not stop that grep, which still
            # matches this table through the word "sigma". Only a `^RESULT: `
            # grep in the driver does that.
            "THE DESIGN THE STUDY ACTUALLY RAN is one run per condition with "
            "sigma imported from this",
            f"table, which detects {mde_external_sigma(sd, 1):.4f} in alpha. "
            f"Everything below is what REPLICATING would buy.",
            "",
            "     N   sigma from the two arms   sigma from this floor   "
            "floor known to within"]
    for n in (2, 3, 4, 5, 6, 8, 10):
        df = cells * (n - 1)
        rows.append(f"    {n:2d}           {mde_two_sample(sd, n):.4f}"
                    f"                 {mde_external_sigma(sd, n):.4f}"
                    f"              {sd_upper_bound(1.0, df):.2f}x  ({df} df)")
    rows.append("")
    rows.append("  effect the study wants to claim        size    replicates needed")
    for effect in EFFECTS:
        need = replicates_for(effect.size, sd)
        label = effect.name + (f" [{effect.model}]" if effect.model else "")
        rows.append(f"    {label:<34.34s}{effect.size:.4f}   "
                    + ("out of reach under 500" if need is None else f"N = {need}"))
    rows.append("")
    rows.append(f"  N = {DEFAULT_REPLICATES} is NOT set by power -- power alone is "
                f"satisfied at N = 2 for both big effects. It is set by the last "
                f"column: below")
    rows.append("  N = 6 the floor is known to worse than 1.5x, and a floor quoted "
                "to a factor of three is not a floor.")
    return "\n".join(rows)


def render_control(control: dict[str, PairedDifference],
                   cards: dict[str, PairedDifference]) -> str:
    ctrl = control[PRIMARY_FIELD]
    card = cards[PRIMARY_FIELD]
    out = ["## The num_stages control, and the comparison it scores", "",
           "Both are paired over the SAME matched cells. Neither needed a GPU and "
           "neither has ever been run before.", ""]
    out.append(f"CONTROL   {ctrl.first}")
    out.append(f"       -> {ctrl.second}")
    out.append(f"          varied: {', '.join(ctrl.varied) or 'nothing detected'}"
               f"   same machine: {ctrl.same_machine}")
    for name in ALPHA_FIELDS:
        out.append("          " + control[name].line())
    out.append("")
    out.append(f"UNDER TEST {card.first}")
    out.append(f"        -> {card.second}")
    out.append(f"          varied: {', '.join(card.varied) or 'nothing detected'}"
               f"   same machine: {card.same_machine}")
    for name in ALPHA_FIELDS:
        out.append("          " + cards[name].line())
    out.append("")
    if (ctrl.mean is not None and card.mean is not None
            and ctrl.mde is not None):
        out.append(f"READ IT: a scheduling knob on ONE card, at one L2, moves "
                   f"{PRIMARY_FIELD} by {abs(ctrl.mean):.4f}. A 1.5x change in L2 "
                   f"capacity moves it by {abs(card.mean):.4f}. The card exceeds "
                   f"the control by {abs(card.mean) - abs(ctrl.mean):+.4f}, "
                   f"against the control's own detection limit of {ctrl.mde:.4f} "
                   f"-- a gap {ctrl.mde / max(1e-12, abs(abs(card.mean) - abs(ctrl.mean))):.0f}x "
                   f"smaller than the smallest gap this design could have seen.")
        out.append("         BOTH differences are themselves under their own "
                   "detection limits, so the honest statement is not 'the cards "
                   "agree' but 'this design cannot tell the cards apart, and it "
                   "cannot tell a card apart from a pipeline stage either'.")
    if card.sd is not None and card.sd > 0 and card.mde is not None:
        need = cells_for(CROSS_CARD_EFFECT.size, card.sd)
        out.append(f"POWER:   at sd_d = {card.sd:.4f} the {card.n}-cell paired "
                   f"design detects {card.mde:.4f}. Detecting the observed "
                   f"{abs(card.mean):.4f} needs "
                   + ("more than 5000" if need is None else f"{need}")
                   + " matched cells, which is "
                   + ("" if need is None else
                      f"{need / card.n:.1f}x the cells actually run, i.e. "
                      f"{math.sqrt(need / card.n):.1f}x in standard error."))
    return "\n".join(out)


def render_mde_line(n: int, arms: list[Arm]) -> list[str]:
    """The one line B14 says every arm's plan must print, and its assumption.

    An MDE with no stated sigma is a number with no units. The sigma here is
    `prior_sd()`, the s3-vs-s4 proxy over sqrt(2), and its scope is on the line
    beside it because it is an upper bound on SAME-SESSION rerun noise and on
    nothing else. Two limits are printed because the run has two designs in it:
    the replicate design (n per condition) and the design every published
    difference in this repo actually is (one run per condition, sigma imported).
    """
    per_model = ", ".join(
        f"{m} {sd:.4f}" for m, sd in sorted(PRIOR_SD_BY_MODEL.items()))
    out = [
        f"MDE: at the assumed sigma {prior_sd():.4f} this plan resolves "
        f"{mde_two_sample(prior_sd(), n):.4f} in alpha with N={n} replicates per "
        f"condition, and {mde_external_sigma(prior_sd(), 1):.4f} at the one-run "
        f"design the study published.",
        f"     the sigma is ASSUMED, not measured here: {PRIOR_SD_SOURCE}",
        f"     it is heteroscedastic by model ({per_model}), so a per-cell "
        f"comparison at mixtral G>=8 is against a wider spread than the pool.",
    ]
    for arm in sorted({a.model for a in arms}):
        for effect in EFFECTS:
            if effect.model != arm:
                continue
            limit = mde_two_sample(prior_sd(), n)
            verdict = ("ABOVE the limit, so this design can see it"
                       if effect.size >= limit else
                       "BELOW the limit: this design CANNOT see it, and the "
                       "gate for it is registered as an expected FAIL")
            out.append(f"     {effect.name} on {arm} is {effect.size:.4f}, "
                       f"{verdict}.")
    return out


def render_plan(arms: list[Arm], n: int, cache_mode: str, base: Path,
                gpu_name: str, costs: dict[str, float | None],
                order: str, sweep_args: Sequence[str], python: str) -> str:
    """The plan, including the id every replicate will resume into.

    `sweep_args`, `order` and `python` are here so the PRINTED ids are the ids
    the run will use. The plan is what an operator reads before spending a pod
    hour, and a plan that names a directory the run then does not use is worse
    than no plan: it is the one artefact that would have shown the collision.

    THAT INCLUDES THE `synthetic-` PREFIX, and it did not until 2026-09-02. The
    commit that made a rehearsal write under the prefix took it in
    `run_replicate` alone, so every rehearsal id printed here was wrong by
    exactly the prefix that commit added: the plan named directories that were
    never written. Both callers now go through `synthetic_run_id`, which is why
    it is a function rather than two lines that have to be kept the same.
    """
    out = ["## The plan", "",
           f"{len(arms)} arm(s) x {n} replicates = {len(arms) * n} sweep processes, "
           f"cache mode {cache_mode}, config device {gpu_name}",
           f"EVERYTHING IS SAVED TO  {base}", ""]
    out += render_mde_line(n, arms)
    out.append("")
    out.append("INTERLEAVED. The two swizzles of one model run back to back "
               "inside a replicate, so their contrast is paired within minutes "
               "rather than across blocks:")
    out += ["  " + line for line in order_lines(arms, n, order)]
    out.append("")
    out.append("SCOPE of any floor this produces: one card, one session, "
               f"models {', '.join(sorted({a.model for a in arms}))}; "
               f"BLOCK_SIZE_N {sorted({a.block_n for a in arms})}, num_stages "
               f"{sorted({a.num_stages for a in arms})}, dtype "
               f"{sorted({a.dtype for a in arms})}, seed held FIXED at "
               f"{sorted({a.seed for a in arms})}, instrument "
               f"{timing_basis()!r}. It bounds nothing outside that list, and "
               "cross-pod noise least of all.")
    out.append("")
    total_model = 0.0
    unknown = False
    for arm in arms:
        secs = costs.get(arm.name)
        if secs is None:
            unknown = True
            cost = "cost UNKNOWN (the sweep's dry run did not answer)"
        else:
            total_model += secs * n
            cost = (f"{secs:.0f} s modelled, ~{secs * WALL_OVER_MODEL:.0f} s wall "
                    f"x {n} = ~{secs * WALL_OVER_MODEL * n / 60:.1f} min")
        out.append(f"  {arm.name:<14s} {arm.model} {arm.dtype} G={arm.group_m} "
                   f"BN={arm.block_n} s={arm.num_stages} r_max={arm.r_max}  {cost}")
        for i in range(1, n + 1):
            run_id = synthetic_run_id(
                run_id_for(arm, i, gpu_name=gpu_name, cache_mode=cache_mode,
                           sweep_args=sweep_args, order=order, python=python),
                sweep_args)
            out.append(f"      rep {i}: {run_id}")
    out.append("")
    if unknown:
        out.append("TOTAL COST: UNKNOWN. At least one arm's dry run gave no "
                   "estimate, and a total assembled from the arms that did answer "
                   "would understate it.")
    else:
        out.append(f"TOTAL: ~{total_model * WALL_OVER_MODEL / 60:.0f} min of GPU "
                   f"({total_model:.0f} s modelled, scaled by the "
                   f"{WALL_OVER_MODEL:.2f}x wall-over-model factor observed on the "
                   f"s4 arm: 127 s logged against 54 s modelled for mixtral_g1).")
    return "\n".join(out)


def render_predictions(n: int, arms: list[Arm]) -> str:
    """The registered expectations.

    NOTHING HERE IS A RESULT AND NOTHING HERE IS SHAPED LIKE ONE. The expected
    verdicts used to print as `[PASS]` and `[FAIL]`, which is exactly the shape
    a scored gate prints, and the session driver's free-text summary grep read
    them out of a REFUSED log as measured output. Expectations are written as
    `expect PASS` here; the only result-shaped line this file emits is
    `moe.bench.exit_codes.result_line`, and a refusal emits none.

    THAT IS NOT ENOUGH BY ITSELF, and the limit is worth naming where the rows
    are built. The driver's regex for this arm is still
    `^[[:space:]]*V[0-9][[:space:]]|floor|sigma`, which catches these rows for
    their leading `V1..V7` and for the word `floor` whatever verdict word they
    carry, so a refusal's registered expectations still reach the session
    summary. What the wording buys is that nothing here can be MISTAKEN for a
    measurement by a reader or parsed as one; keeping it out of the summary
    altogether is the driver's regex to change.
    """
    per_model = []
    for model, group in swizzle_pairs(arms).items():
        registered = next((e for e in EFFECTS
                           if e.model == model and e.name == "swizzle swing"),
                          None)
        size = "unregistered" if registered is None else f"{registered.size:.4f}"
        want = "PASS" if registered is not None and registered.size >= 0.10 \
            else "FAIL"
        ends = "->".join(f"G={a.group_m}" for a in group)
        per_model.append(f"  {swizzle_gate_name(model):<34s} expect {want}  "
                         f"{ends}, registered at {size}")
    return f"""\
## Predictions, registered before anything ran

VALIDITY -- a FAIL means no number from part (a) may be quoted.
  V1  every replicate got its own run id and directory   distinct ids == {n} x arms
  V2  the planned replicates ran and produced fits       >= 2 cells at n = {n}
  V3  no cell returned identical alpha every time        zero degenerate cells
  V4  the Triton cache behaved as the mode claims        fresh: >= 1 artefact each
  V5  the cells share a spread                           max sd / min sd <= {HOMOGENEITY_RATIO:g}
  V6  one instrument across every replicate              instrument == {timing_basis()!r}
  V7  the floor covers more than one model               >= 2 models, or --single-model-floor

CLAIM -- a FAIL is a result, not a broken run.
  C1 floor size                       expect PASS  sd <= {prior_sd():.4f}
  C2 cross-card under floor           expect PASS  |{CROSS_CARD_EFFECT.size:.4f}| < MDE
{chr(10).join(per_model)}
  C4 card beats stages control        expect FAIL  |card| - |stages| >= MDE
  C5 sign consistency                 expect FAIL  one sign across three fields

C4 and C5 are registered as EXPECTED FAILURES from the committed reports alone, and
both are computed here without a GPU. C4 fails because the same 11 cells move
+0.0101 when num_stages goes 4 -> 3 on one card and +0.0117 when the card changes.
C5 fails because the cross-card difference is +0.0117 in alpha-corrected and
-0.3325 in alpha-upper -- two anchorings of the same fit, opposite signs -- while
the num_stages control keeps one sign across all three.

THE QWEN2 C3 IS THE THIRD REGISTERED FAILURE and it is the reason that arm is in
the plan. Its swizzle effect is 0.0226 at its largest committed H200 cell against
an N={n} detection limit of {mde_two_sample(prior_sd(), n):.4f}, and on the A100 it
changes SIGN between BLOCK_M 32 and 64. Expecting it to fail means the replicate
arm cannot come back having confirmed a swizzle mechanism on the one model where
it is twelve sigma and silently said nothing about the model where it is absent.
Registering all three now means none can be reported later as a discovery."""


# --------------------------------------------------------------------------
# running the replicates
# --------------------------------------------------------------------------

DRY_COST_RE = re.compile(r"estimated GPU time\s+([0-9.]+)\s*s")


#: The exit codes a `--dry-run` cost probe is allowed to come back with. REFUSED
#: is the one the repository's dry-run census settled on and the one the sweep
#: has returned since 2026-09-02: a plan measured nothing, so it scores no gate
#: and its own log classifies as a refusal. DONE stays accepted because it is
#: what the sweep returned before that date and this probe is also run against
#: checkouts and logs from before it. Anything else -- CLAIM_FAIL, INVALID,
#: ERROR -- is a probe that broke rather than a plan that priced itself, and the
#: caller must not read a cost out of it. An argparse failure also exits 2 and is
#: therefore inside this tuple; it is caught one line later instead, because
#: argparse writes its usage to STDERR and leaves stdout empty, so `DRY_COST_RE`
#: finds no cost and the function still returns None.
PLAN_CODES = (exit_codes.DONE, exit_codes.REFUSED)


def sweep_cost(arm: Arm, python: str) -> float | None:
    """Ask the sweep itself what one arm costs. None when it will not say.

    None rather than a guess: a fabricated cost on a metered pod is how a
    session runs out of budget three arms from the end.

    IT USED TO KEY ON `returncode != 0` AND THAT WAS A LATENT COUPLING. The
    sweep's `--dry-run` returned DONE only because nobody had moved it yet, and
    the day it moved to REFUSED this function would have started returning None
    for every arm, `render_plan` would have printed "TOTAL COST: UNKNOWN", and
    the whole "TOTAL ... min of GPU" line the pod budget is set from would have
    vanished from the plan. Not a wrong number, a missing line, on a rented pod,
    silently. The probe asks a question about a PLAN, so the codes a plan is
    allowed to exit with are the ones it accepts, and `PLAN_CODES` names them.
    """
    argv = [python, str(SWEEP), "--dry-run"] + arm.sweep_argv("cost-probe", Path("/tmp"))
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode not in PLAN_CODES:
        return None
    found = DRY_COST_RE.search(done.stdout)
    return float(found.group(1)) if found else None


def link_shared_cache(out_dir: Path, run_id: str, shared: Path) -> None:
    """Point one replicate's triton-cache at the shared one, for --warm-cache.

    The sweep hardcodes <out_dir>/block_m_crossing/<run_id>/triton-cache and sets
    TRITON_CACHE_DIR to it before importing vLLM, so the only way to warm the
    cache across replicates is to make that path resolve to a shared directory.
    `Path.mkdir(exist_ok=True)` follows the symlink and succeeds, so the sweep
    needs no change.
    """
    shared.mkdir(parents=True, exist_ok=True)
    target = out_dir / "block_m_crossing" / run_id
    target.mkdir(parents=True, exist_ok=True)
    link = target / "triton-cache"
    # is_symlink() as well as exists(): a DANGLING symlink reports exists() False
    # and symlink_to() then raises FileExistsError, killing the replicate for a
    # reason that has nothing to do with the measurement.
    if not link.is_symlink() and not link.exists():
        link.symlink_to(shared, target_is_directory=True)


def run_replicate(arm: Arm, index: int, base: Path, *, gpu_name: str,
                  cache_mode: str, python: str, extra: list[str],
                  sweep_args: Sequence[str], order: str,
                  shared_cache: Path | None, timeout_s: float) -> Replicate:
    """One sweep process. Its log survives even when it fails.

    TWO ARGUMENT LISTS, ON PURPOSE. `extra` is what goes on the child's command
    line; `sweep_args` is the part of it that NAMES the run. They differ only
    under `--rehearse`, which appends a per-replicate `--seed` so the synthetic
    cells move: that seed is a function of `index`, which is already in the key,
    so putting it in as well would make the ids the plan printed disagree with
    the ids the run used, and the plan is the artefact an operator reads before
    spending a pod hour. Everything an operator can vary IS in `sweep_args`,
    because we hand the child our `--run-id` and it therefore cannot separate
    two settings we did not separate for it.

    THE PREFIX IS APPLIED HERE AND IN `render_plan`, AND THE TWO MUST AGREE.
    `synthetic_run_id` is the one place that decides, because when only this
    half took the prefix the plan printed ids for directories that were never
    written, which is a log and a process saying two different things about one
    run.

    THE CHILD'S EXIT CODE IS READ THROUGH `REPORT_CODES`, NOT AGAINST 0. A
    child that measured every cell and then failed its own CLAIM gate exits 1
    and its `report.json` is exactly as good as a child that exited 0; see
    `REPORT_CODES` for the two GPU hours that cost. `Replicate.ok` reads the
    same tuple, and it has to be the same tuple: those two places disagreeing
    is the bug, not the rule.
    """
    run_id = synthetic_run_id(
        run_id_for(arm, index, gpu_name=gpu_name, cache_mode=cache_mode,
                   sweep_args=sweep_args, order=order, python=python),
        extra)
    out_dir = base / f"{arm.name}-rep{index}"
    rep = Replicate(arm=arm.name, index=index, run_id=run_id, out_dir=out_dir,
                    report=out_dir / "block_m_crossing" / run_id / "report.json")
    out_dir.mkdir(parents=True, exist_ok=True)
    if shared_cache is not None:
        link_shared_cache(out_dir, run_id, shared_cache)
    argv = [python, str(SWEEP)] + arm.sweep_argv(run_id, out_dir) + extra
    log = base / "logs" / f"{arm.name}-rep{index}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        done = subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout_s)
        rep.returncode = done.returncode
        log.write_text(done.stdout + "\n--- stderr ---\n" + done.stderr)
        if done.returncode not in REPORT_CODES:
            rep.error = (f"sweep exited {exit_codes.describe(done.returncode)}; "
                         f"no cell of it is pooled; see {log}")
    except subprocess.TimeoutExpired:
        # A hung replicate on a metered pod costs the whole session. The
        # replicate is lost and named; the ones already on disk survive, and V2
        # will refuse to publish a floor built from fewer than the planned N.
        rep.error = (f"sweep exceeded --replicate-timeout {timeout_s:.0f} s and "
                     f"was killed; see {log}")
    except (OSError, subprocess.SubprocessError) as exc:
        rep.error = f"{type(exc).__name__}: {exc}"
    rep.seconds = time.time() - started
    if not rep.error:
        load_replicate(rep)
    return rep


#: The card label every path, run id and published field carries when no device
#: is attached and the operator named none. The same sentinel
#: `block_m_crossing_sweep.NO_CARD_SLUG`, `tile_cap_test`, `bn_decomposition`,
#: `occupancy_vs_swizzle` and the session driver already use, so one `ls` over a
#: shared network volume sorts every unnamed run into one place.
#:
#: IT USED TO BE THE BARE LITERAL "NVIDIA H200". `gpu_name` was
#: `args.gpu_name or device or "NVIDIA H200"`, so a laptop that named no card
#: wrote `results/.../nvidia_h200-fresh-n6/` and stamped `gpu_name: "NVIDIA
#: H200"` into every run id and into the published JSON, indistinguishable in
#: `ls`, in the id and in the file from a run on the rented card. The session
#: driver's own rule, printed at every start, is that every path carries the
#: card or the literal `nocard`; this file was the one that quietly did not.
#: `--gpu-name` still names a card the operator is about to rent, because
#: pricing a plan for it is the supported and useful case: it is a statement
#: now rather than a silent default.
NO_CARD = "nocard"


def resolve_card(named: str, device: str, missing: Sequence[str]) -> tuple[str, str]:
    """`(the card every artefact of this run is labelled with, why)`.

    Falling order of directness: `--gpu-name`, the attached device, then
    `NO_CARD`. Never a guess, because a wrong card here is silent everywhere it
    lands, and the reason travels with the label so the plan can print it.
    """
    if named:
        return str(named), "named by --gpu-name"
    if device:
        return str(device), "read from the attached device"
    why = "; ".join(missing) if missing else "no device and no --gpu-name"
    return NO_CARD, f"no card could be named: {why}"


def detect_gpu() -> tuple[str, list[str]]:
    """(device name, reasons it cannot be measured here)."""
    missing: list[str] = []
    try:
        import torch
    except ImportError as exc:
        return "", [f"torch is not importable: {exc}"]
    if not torch.cuda.is_available():
        missing.append("no CUDA device (torch.cuda.is_available() is False)")
        return "", missing
    return torch.cuda.get_device_name(0), missing


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def default_out_base() -> Path:
    env = os.environ.get("MOE_RESULTS_DIR")
    if env:
        return Path(env) / "replicate_noise_floor"
    if Path("/workspace").is_dir():
        return Path("/workspace/results/replicate_noise_floor")
    return ROOT / "results" / "replicate_noise_floor"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES,
                    help=f"identical repeats per arm; {DEFAULT_REPLICATES} is "
                         f"argued in the module docstring and anything below 3 "
                         f"cannot bound its own sd")
    ap.add_argument("--arms", default=DEFAULT_ARM_NAMES,
                    help="comma list from " + ",".join(a.name for a in DEFAULT_ARMS)
                         + ". Both models by default: a floor measured only on "
                           "mixtral is a floor measured where the swizzle "
                           "effect is 0.3855, and the surface it licenses "
                           "spans models where it is 0.02")
    ap.add_argument("--single-model-floor", action="store_true",
                    help="allow --arms to name one model only. V7 REFUSES a "
                         "single-model floor without this, because the "
                         "consequence -- a mechanism confirmed on the one "
                         "model where it is twelve sigma -- is invisible in "
                         "the output otherwise")
    ap.add_argument("--order", choices=ORDER_MODES, default=ORDER_COUNTERBALANCED,
                    help="how the replicates interleave. counterbalanced (the "
                         "default) alternates the order of each model's "
                         "G=1/G=16 pair across replicates, so a linear drift "
                         "inside a pair cancels out of the mean delta; paired "
                         "keeps G=1 first every time. Both pair the contrast "
                         "within minutes, which the retired blocked design did "
                         "not")
    ap.add_argument("--warm-cache", action="store_true",
                    help="share ONE Triton cache across replicates. Measures the "
                         "narrower, execution-only floor; never the published one "
                         "unless --floor-from warm is also given")
    ap.add_argument("--floor-from", choices=("fresh", "warm"), default="fresh",
                    help="which cache mode may be published as THE floor")
    ap.add_argument("--gpu-name", default=None,
                    help="override the device name used in run ids; off a GPU "
                         "this is what the plan is built for")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--replicate-timeout", type=float, default=1800.0,
                    help="kill a single sweep process after this many seconds. "
                         "The default is roughly 14x the 127 s the mixtral arm "
                         "took on the s4 run, so it fires only on a hang")
    ap.add_argument("--sweep-arg", action="append", default=[],
                    help="extra argument passed to every sweep process, repeatable")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, the cost and the predictions, run part "
                         "(b) in full, measure nothing")
    ap.add_argument("--control-only", action="store_true",
                    help="part (b) only: the num_stages control from committed "
                         "reports, no GPU")
    ap.add_argument("--rehearse", type=float, default=None, metavar="ALPHA",
                    help="run the replicates through the sweep's --self-test at "
                         "this alpha, varying only the synthetic seed. Exercises "
                         "every line of the plumbing off GPU and is stamped "
                         "SYNTHETIC everywhere; noise_floor() refuses it")
    ap.add_argument("--rehearse-noise", type=float, default=0.02,
                    help="lognormal sigma handed to the sweep's --self-test-noise")
    ap.add_argument("--publish", action="store_true",
                    help=f"write {NOISE_FLOOR_JSON}")
    return ap


def resolve_arms(names: str) -> list[Arm]:
    known = {a.name: a for a in DEFAULT_ARMS}
    wanted = [n for n in names.split(",") if n]
    unknown = [n for n in wanted if n not in known]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; known arms are {sorted(known)}")
    if not wanted:
        raise SystemExit("--arms must name at least one arm")
    return [known[n] for n in wanted]


def _main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    arms = resolve_arms(args.arms)
    cache_mode = "warm" if args.warm_cache else "fresh"
    n = args.replicates
    if n < 2:
        raise SystemExit("--replicates below 2 cannot produce a standard deviation")

    control = {f: stages_control(f) for f in ALPHA_FIELDS}
    cards = {f: cross_card(f) for f in ALPHA_FIELDS}

    print("# The noise floor this study never measured")
    print()
    print(SIGN_BANNER)
    print()
    print(render_control(control, cards))
    print()
    print(render_power_table(prior_sd(), "the s3/s4 proxy, an UPPER bound",
                             cells=CELLS_PER_ARM * len(arms)))

    if args.control_only:
        print()
        print("=" * 72)
        print("PART (b) ONLY. The replicate floor was not measured and no number "
              "on this page is one.")
        print("  Nothing above is a scored gate: this page prints no RESULT "
              "line, so `exit_codes.classify_text` reads it as nothing scored, "
              "which is what REFUSED means.")
        print("=" * 72)
        if args.publish:
            print(write_published(build_document(control, cards, None,
                                                unmeasured_provenance())))
        print()
        print(IMPORT_BANNER)
        return exit_codes.REFUSED

    device, missing = detect_gpu()
    gpu_name, card_reason = resolve_card(args.gpu_name, device, missing)
    # THE CARD IS IN THE BASE DIRECTORY (2026-09-02). It was `{cache_mode}-n{n}`,
    # which names no machine, and the per-replicate directory below it is
    # `{arm}-rep{i}`, which names none either -- so two pods writing to the same
    # network volume, which is what `$MOE_RESULTS_DIR` and `/workspace/results`
    # ARE FOR, interleaved their logs/ trees and only opening a file said which
    # card wrote it. Collision 2 of `moe.bench.provenance`'s docstring.
    base = ((args.out_dir or default_out_base())
            / f"{PV.card_slug(gpu_name)}-{cache_mode}-n{n}")
    print()
    print(f"card     {gpu_name} -- {card_reason}")
    if gpu_name == NO_CARD:
        print(f"         Every path, run id and published field below carries "
              f"the literal '{NO_CARD}', so nothing this run writes can be "
              f"mistaken in `ls` for a run on a real card. Pass --gpu-name to "
              f"price a plan for a card you are about to rent.")
    costs = {arm.name: sweep_cost(arm, args.python) for arm in arms}

    rehearsing = args.rehearse is not None
    # THE PASSTHROUGH IS ASSEMBLED BEFORE THE PLAN IS PRINTED, because it is part
    # of every replicate's run id now and a plan that printed ids the run would
    # not use is exactly the artefact that should have caught this.
    extra: list[str] = list(args.sweep_arg)
    if rehearsing:
        extra += ["--self-test", str(args.rehearse),
                  "--self-test-noise", str(args.rehearse_noise)]

    print()
    print(render_plan(arms, n, cache_mode, base, gpu_name, costs, args.order, extra,
                      args.python))
    print()
    print(render_predictions(n, arms))

    blocked = args.dry_run or (bool(missing) and not rehearsing)
    if blocked:
        why = "--dry-run was given" if args.dry_run else "; ".join(missing)
        print()
        print("=" * 72)
        print("NOT A RESULT. The replicate floor was not measured.")
        print(f"  reason: {why}")
        print("  Part (b) above IS a result: it is arithmetic over committed")
        print("  reports and needed no GPU. Part (a) needs one.")
        print(f"  On the pod:  {Path(sys.argv[0]).name} --replicates {n} --publish")
        print("  No RESULT line was printed, because nothing was scored.")
        print("=" * 72)
        if args.publish:
            print(write_published(build_document(control, cards, None,
                                                unmeasured_provenance())))
        print()
        print(IMPORT_BANNER)
        return exit_codes.REFUSED

    shared_cache = (base / "shared-triton-cache") if args.warm_cache else None
    if rehearsing:
        print()
        print("REHEARSAL: every replicate below runs the sweep's --self-test, so "
              "its cells are GENERATED from the model and nothing is measured. "
              "The seed varies per replicate purely to make the plumbing move; a "
              "real replicate holds the seed FIXED. Stamped synthetic everywhere.")
        print("  C1 and every C3 are MEANINGLESS here and their verdicts must be "
              "ignored: the floor is whatever --rehearse-noise was set to, and "
              "the sweep's synthetic cells do not depend on GROUP_SIZE_M, so the "
              "two arms of a model are the same data and C3 reads a swizzle "
              "delta of exactly 0.0000. That C3 catches it is the point of "
              "running this.")
        print("  V6 WILL FAIL and must: the sweep stamps its self-test cells "
              "with the synthetic instrument, not with the one this repo "
              "publishes under. That is the gate working, not the plumbing "
              "breaking.")
        print("  AND THE EXIT CODE DOES NOT DEPEND ON IT. A rehearsal is "
              "forced to INVALID by `rehearsal_exit` whatever the gates say, "
              "because since the parent started reading its children's reports "
              "a rehearsal genuinely scores and was one instrument stamp away "
              "from printing `exit 0 DONE` over a generated floor.")

    base.mkdir(parents=True, exist_ok=True)
    replicates: list[Replicate] = []
    started = time.time()
    # INTERLEAVED, and the order is the one the plan printed. `run_order`
    # alternates each model's G=1/G=16 pair across replicates, so the swizzle
    # contrast is paired within minutes and a linear drift inside a pair
    # cancels out of the mean rather than being added to every delta. The
    # retired loop ran all N of one arm and then all N of the other.
    schedule = run_order(arms, n, args.order)
    for position, (arm, index) in enumerate(schedule, start=1):
        per_rep = list(extra)
        if rehearsing:
            # The sweep's synthetic cells are a deterministic function of the
            # seed, so a rehearsal that held it fixed would produce N identical
            # reports, sd 0.0000, and would exercise nothing except the V3
            # gate. A real replicate does the opposite: seed FIXED.
            per_rep += ["--seed", str(1000 + index)]
        print(f"  [{position}/{len(schedule)}] {arm.name} rep {index}/{n} "
              f"launching", flush=True)
        rep = run_replicate(arm, index, base, gpu_name=gpu_name,
                            cache_mode=cache_mode, python=args.python,
                            extra=per_rep, sweep_args=extra, order=args.order,
                            shared_cache=shared_cache,
                            timeout_s=args.replicate_timeout)
        replicates.append(rep)
        if not rep.ok:
            status = f"FAILED: {rep.error}"
        elif rep.returncode == exit_codes.DONE:
            status = "ok"
        else:
            # Pooled, and named. The child measured soundly and its own claim
            # gate refuted it, which says nothing about the spread this parent
            # is here to measure; printing a bare "ok" would hide from the
            # operator that the sweep disagreed with itself on a rented card.
            status = (f"ok ({exit_codes.CODE_NAMES[rep.returncode]}: the sweep's "
                      f"own claim gate did not pass; its cells are measured and "
                      f"are pooled)")
        print(f"      {rep.seconds:.0f} s  {status}", flush=True)

    floors = {f: pool(spreads_for(replicates, f), f) for f in ALPHA_FIELDS}
    primary_spreads = spreads_for(replicates, PRIMARY_FIELD)

    gates = validity_gates(replicates, n, arms, cache_mode, floors,
                           single_model_ok=args.single_model_floor)
    gates += claim_gates(floors, primary_spreads, control, cards, arms)

    print()
    print("## Between-replicate spread, per cell")
    print()
    print(f"  {'arm':<14s}{'BM':>5s}{'n':>4s}{'mean':>10s}{'sd':>10s}{'cv':>9s}")
    for spread in primary_spreads:
        cv = spread.cv
        print(f"  {spread.arm:<14s}{spread.block_m:>5d}{spread.n:>4d}"
              f"{spread.mean:>10.4f}{spread.sd:>10.4f}"
              + (f"{cv:>8.2%}" if cv is not None else "       --"))
    print()
    for name in ALPHA_FIELDS:
        floor = floors[name]
        if floor.pooled_sd is None:
            print(f"  {name:16s} NO FLOOR: {floor.reason}")
        else:
            print(f"  {name:16s} between-replicate sd {floor.pooled_sd:.4f} on "
                  f"{floor.df} df, upper 95% {floor.upper95:.4f} -- "
                  f"{floor.reason}")

    print()
    print("## The instrument every replicate was timed by")
    print()
    for rep in replicates:
        state = rep.timing_state or {}
        print(f"  {rep.arm:<14s} rep {rep.index}  "
              f"{state.get('instrument') or '<none stamped>'}  "
              f"warmup {state.get('warmup_ms')} ms  "
              f"target {state.get('target_ms')} ms  "
              f"clock-excluded cells "
              f"{state.get('cells_excluded_for_clock_level')}")

    primary = floors[PRIMARY_FIELD]
    if primary.pooled_sd:
        measured_cells = len([s for s in primary.spreads if s.n >= 2])
        print()
        # The MEASURED table's df column must use the cells that actually came
        # back, not the CELLS_PER_ARM expectation: an arm that produced one
        # identifiable fit instead of two has half the df and a wider bound, and
        # printing the planned number would hide that.
        print(render_power_table(primary.pooled_sd, "MEASURED",
                                 cells=max(1, measured_cells)))

    print()
    print("## Gates")
    print()
    print(render_gates(gates))

    provenance = (f"{len(arms)} arm(s) x {n} replicates on {gpu_name}, cache "
                  f"{cache_mode}, order {args.order}, "
                  f"{time.time() - started:.0f} s wall, under {base}")
    scope = scope_block(arms, n_replicates=n, cache_mode=cache_mode,
                        gpu_name=gpu_name, order=args.order,
                        single_model_ok=args.single_model_floor)
    prov = PV.provenance_block(
        instrument=timing_basis(),
        warmup_ms=statistics.fmean([a.warmup_ms for a in arms]),
        iters=None,
        target_ms=statistics.fmean([a.cell_budget_ms for a in arms]))
    publishable = cache_mode == args.floor_from
    doc = build_document(
        control, cards, floors if publishable else None, prov,
        n_replicates=n, cache_mode=cache_mode, gpu_name=gpu_name,
        provenance=provenance, synthetic=rehearsing, scope=scope)
    (base / "noise_floor.json").write_text(json.dumps(doc, indent=2) + "\n")
    print()
    print(f"EVERYTHING IS SAVED TO {base}")
    print(f"  summary {base / 'noise_floor.json'}")
    if not publishable:
        print(f"  the {cache_mode} floor is NOT publishable as THE floor "
              f"(--floor-from is {args.floor_from}); replicate_floor is null above")
    gates_allow, gates_refusing = gates_permit_publishing(
        gates, floor_withheld=not publishable, cache_mode=cache_mode,
        floor_from=args.floor_from)
    if args.publish and rehearsing:
        # A synthetic floor in the curated directory would be readable by anyone
        # who passed allow_synthetic, and would sit in git looking exactly like a
        # measurement. --rehearse and --publish are mutually exclusive on purpose.
        print(f"  REFUSING --publish: this was a REHEARSAL. {NOISE_FLOOR_JSON} "
              f"only ever holds numbers measured on a card. The synthetic summary "
              f"is under {base} and nothing else was written.")
    elif args.publish and not gates_allow:
        # The exit code below is already INVALID here. This is the same verdict
        # applied to the artefact: see `gates_permit_publishing` for the two
        # hours of pod time that the log and git disagreeing would have cost.
        print(f"  REFUSING --publish: {gates_refusing} Nothing may land in "
              f"{NOISE_FLOOR_JSON}, which every other script imports as THE "
              f"floor; the full summary is under {base} and nothing in "
              f"results/published was touched.")
    elif args.publish:
        print(write_published(doc))
    else:
        print(f"  --publish would write {NOISE_FLOOR_JSON}")

    print()
    print(IMPORT_BANNER)

    # ONE EXIT CODE, FROM THE SHARED TABLE, OVER THE SAME GATES THAT PRINTED
    # THEIR RESULT LINES. `classify` scores UNKNOWN against the gate: a VALIDITY
    # gate that could not decide leaves the page as unquotable as a FAIL, and a
    # CLAIM gate that could not decide has not established its claim. The
    # driver can recompute this from the log with `classify_text` and a
    # disagreement between the two is itself a defect.
    rc = exit_codes.classify(g.scored() for g in gates)
    print()
    if rehearsing:
        rc, why = rehearsal_exit(rc, gates)
        print(why)
    print(f"exit     {exit_codes.describe(rc)}")
    return rc


def main(argv: list[str] | None = None) -> int:
    """The exit-code table held for the CLI AND for a caller of `main`.

    AN UNPLANNED CRASH IS ERROR (4). Left to propagate, an exception exits the
    interpreter ONE, and ONE is `CLAIM_FAIL`, which `moe/bench/exit_codes.py`
    defines as a RESULT: it is a finished code, the session driver latches the
    arm, `arm()` skips the row on every resume, RETRY_ARMS stays 0 and the
    session exits 0. This arm books 120 minutes and pools its children at the
    very end, so a torch OOM or a truncated report in the last minute of it
    would have been filed as one of this experiment's registered findings and
    never rerun. Measured on 2026-09-03 by injecting `raise RuntimeError` at
    the top of `_main` in a shadow tree: the exit code was 1. Every other arm
    in the session (`bm128_roofline`, `occupancy_vs_swizzle`,
    `calibrate_hardware`) already installed this handler; this file and
    `bm128_depth` did not, and this is the half of the pair that owns the
    session's most expensive arm.

    A STRING `SystemExit` IS A REFUSAL, and that is the second half of the same
    defect. `raise SystemExit("...")` exits ONE as well, and this file has five
    of them: `--replicates` below 2, two in `resolve_arms`, and both walls in
    `write_published`. The last of those can fire AFTER the card has been paid
    for, so it is the one that mattered: a tracked path git would drop, or a
    null document over a measured floor, exited CLAIM_FAIL and the driver
    latched a 120-minute arm as finished. REFUSED (2) is the table's word for
    "a precondition was not met and nothing was published"; the sentence goes
    to stderr first, because a code with no sentence tells an operator nothing
    to fix, and the run's full summary is under its own base directory either
    way.

    Caught here rather than at the raise sites so that a refusal added later
    cannot reintroduce the bug by forgetting the code, and so the contract is
    the same one whether this file is run or imported.
    """
    try:
        return _main(argv)
    except SystemExit as exc:
        if isinstance(exc.code, str):
            msg = (exc.code if exc.code.startswith("REFUS")
                   else f"REFUSED: {exc.code}")
            print(msg, file=sys.stderr)
            return exit_codes.REFUSED
        raise
    except Exception:                                     # noqa: BLE001
        traceback.print_exc()
        print("ERROR: replicate_noise_floor crashed before it could reach a "
              "verdict. This is the apparatus failing, not a claim failing, so "
              f"it exits {exit_codes.ERROR} and not {exit_codes.CLAIM_FAIL}: "
              "the traceback above is the thing to fix, the replicates already "
              "on disk are under the base directory the plan printed, and the "
              "arm may be re-run.", file=sys.stderr)
        return exit_codes.ERROR


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
