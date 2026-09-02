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

Only (3) is exchangeable with the comparison being scored. The cross-card arms
ran in separate processes on separate pods with separate caches; so did s3 and
s4. A floor measured at (1) or (2) would be narrower than the noise actually
present in every difference the study reports, and scoring those differences
against it is the exact mechanism by which a null becomes a finding. So a
replicate here is a separate `block_m_crossing_sweep.py` PROCESS with its own
output directory, which gives it its own Triton cache for free -- the sweep
already points TRITON_CACHE_DIR at <out_dir>/triton-cache before it imports vLLM.

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
proxy: paired sd 0.0323 over 11 cells, so a per-arm sd of 0.0323/sqrt(2) =
0.0228. That proxy CONFOUNDS num_stages, so it is an UPPER bound on replicate
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
cell, and saying so is the point rather than a limitation.

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

EXIT CODES. 0 every gate passed, 1 a gate FAILED, 3 nothing was measured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SWEEP = ROOT / "scripts" / "block_m_crossing_sweep.py"
PUBLISHED = ROOT / "results" / "published"

#: Where the importable number lives. `results/*` is ignored with only
#: `!results/published/` excepted, so this is one of the few paths under
#: `results/` git will take, and `git_accepts()` below checks it at write time
#: rather than trusting this comment.
NOISE_FLOOR_JSON = PUBLISHED / "NOISE_FLOOR.json"

SCHEMA = "moe-kernels/noise-floor/1"

EXIT_OK, EXIT_GATE_FAILED, EXIT_NOT_MEASURED = 0, 1, 3

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


EFFECTS: tuple[Effect, ...] = (
    Effect("swizzle swing", 0.3855,
           "s4 arm, mixtral BLOCK_M=32: alpha-corrected 1.0073 at GROUP_SIZE_M=1 "
           "vs 0.6218 at 16"),
    Effect("footprint spread at G=1", 0.4110,
           "s4 arm, all seven GROUP_SIZE_M=1 fits: 1.0073 (mixtral BN=64 BM=32) "
           "down to 0.5963 (deepseek-v2-lite BM=32)"),
    Effect("cross-card L2", 0.0117,
           "A100 s3 minus H200 s3, 11 matched cells, paired mean of "
           "alpha-corrected"),
)

#: The best prior estimate of per-arm replicate sd, and its provenance. The
#: s3-vs-s4 paired sd is 0.0323 over 11 cells; dividing by sqrt(2) turns a
#: difference-of-two sd into a per-arm sd. It CONFOUNDS num_stages with rerun
#: noise, so it can only be an upper bound, which is what makes it safe to size N
#: against.
PRIOR_SD = 0.0323 / math.sqrt(2.0)

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
    sigma of 0.0228 the limit is 0.0903, still 7.7x the cross-card difference the
    study reported.
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

    The defaults are the two ends of the swizzle swing on the model whose arm is
    cheapest, so the same replicates that measure the floor also score the
    largest effect the study claims -- with replicates, which no comparison in
    this repo has ever had.
    """

    name: str
    model: str = "mixtral-8x7b"
    dtype: str = "bf16"
    group_m: int = 1
    block_n: int = 64
    num_stages: int = 4
    tiles: str = "32,64,128,256"
    r_max: int = 1024
    row_step: int = 32
    step_probes: int = 6
    iters: int = 50
    warmup: int = 20
    cell_budget_ms: float = 400.0
    seed: int = 0

    def sweep_argv(self, run_id: str, out_dir: Path) -> list[str]:
        return [
            "--model", self.model, "--dtype", self.dtype, "--tiles", self.tiles,
            "--r-max", str(self.r_max), "--row-step", str(self.row_step),
            "--step-probes", str(self.step_probes),
            "--num-stages", str(self.num_stages), "--group-m", str(self.group_m),
            "--block-n", str(self.block_n), "--iters", str(self.iters),
            "--warmup", str(self.warmup),
            "--cell-budget-ms", str(self.cell_budget_ms), "--seed", str(self.seed),
            "--run-id", run_id, "--out", str(out_dir),
        ]


DEFAULT_ARMS: tuple[Arm, ...] = (
    Arm("mixtral_g1", group_m=1),
    Arm("mixtral_g16", group_m=16),
)


def run_id_for(arm: Arm, replicate: int, *, gpu_name: str, cache_mode: str) -> str:
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
    """
    payload = json.dumps({
        "arm": arm.name, "model": arm.model, "dtype": arm.dtype,
        "group_m": arm.group_m, "block_n": arm.block_n,
        "num_stages": arm.num_stages, "tiles": arm.tiles, "r_max": arm.r_max,
        "row_step": arm.row_step, "step_probes": arm.step_probes,
        "iters": arm.iters, "warmup": arm.warmup,
        "cell_budget_ms": arm.cell_budget_ms, "seed": arm.seed,
        "gpu": gpu_name, "cache": cache_mode, "replicate": replicate,
    }, sort_keys=True)
    slug = re.sub(r"[^a-z0-9]+", "", gpu_name.lower())[:12] or "unknowngpu"
    return (f"nf-{arm.name}-{arm.model}-{arm.dtype}-g{arm.group_m}"
            f"-n{arm.block_n}-s{arm.num_stages}-r{arm.r_max}-{cache_mode}"
            f"-{slug}-rep{replicate}-{hashlib.sha1(payload.encode()).hexdigest()[:8]}")


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

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.error and bool(self.cells)


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
    """Read one replicate's report.json into ladder cells."""
    if not rep.report.exists():
        rep.error = f"no report.json at {rep.report}"
        return rep
    doc = json.loads(rep.report.read_text())
    fixed = doc["fixed"]
    rep.compiles = parse_compiles(doc)
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
    counts as a pass.
    """

    name: str
    prediction: str
    rule: str
    expected: str
    passed: bool | None
    observed: str
    invalidates: str = ""

    def render(self) -> str:
        tag = {True: "PASS", False: "FAIL", None: "UNKNOWN"}[self.passed]
        surprise = ""
        if self.passed is not None:
            got = "PASS" if self.passed else "FAIL"
            surprise = "" if got == self.expected else f"  <-- expected {self.expected}"
        out = [f"[{tag}] {self.name}  {self.prediction}{surprise}",
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


def validity_gates(replicates: list[Replicate], expected_n: int, arms: list[Arm],
                   cache_mode: str, floors: dict[str, PooledFloor]) -> list[Gate]:
    """V-gates. A FAIL means no number from part (a) may be quoted."""
    gates: list[Gate] = []
    done = [r for r in replicates if r.ok]

    gates.append(Gate(
        "V1 distinct runs",
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
        "V2 non-vacuity",
        "the replicates that were planned actually ran and produced fits",
        f"{want} replicates complete and at least 2 cells with all "
        f"{expected_n} values present",
        "PASS",
        None if not replicates else (
            len(done) == want
            and len([s for s in floors[PRIMARY_FIELD].spreads
                     if s.n == expected_n]) >= 2),
        f"{len(done)} of {want} replicates ok; "
        f"{len([s for s in floors[PRIMARY_FIELD].spreads if s.n == expected_n])} "
        f"full cells of {len(floors[PRIMARY_FIELD].spreads)} seen"
        if replicates else "nothing was launched",
        "everything below: a check that examined nothing also reports zero "
        "failures"))

    degenerate = [s for s in floors[PRIMARY_FIELD].spreads if s.degenerate]
    gates.append(Gate(
        "V3 not a collision",
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
            "V4 fresh cache",
            "every replicate compiled its own Triton artefacts at every setting",
            "minimum fresh-artefact count over replicates and settings >= 1",
            "PASS",
            None if not counts else min(counts) >= 1,
            "no replicate reported a compile count" if not counts
            else f"minimum {min(counts)} artefacts over {len(counts)} replicates",
            "the unit: a warm cache makes this a narrower floor than the "
            "comparisons it is meant to score"))
    else:
        # Replicates are launched arm-major, so "everything after the first" is
        # NOT a tail slice of the list: it is every replicate whose index is
        # above 1, one per arm being excluded rather than the first len(arms)
        # entries, which would have been reps 1 and 2 of the FIRST arm only.
        later = [min(r.compiles.values()) for r in done if r.index > 1 and r.compiles]
        gates.append(Gate(
            "V4 warm cache",
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
        "V5 homogeneity",
        "the cells share a spread, so pooling them into one floor is legitimate",
        f"max cell sd / min cell sd <= {HOMOGENEITY_RATIO:g} in {PRIMARY_FIELD}",
        "PASS",
        None if PRIMARY_FIELD not in ratios
        else ratios[PRIMARY_FIELD] <= HOMOGENEITY_RATIO,
        "fewer than two cells had a positive spread" if PRIMARY_FIELD not in ratios
        else f"{ratios[PRIMARY_FIELD]:.2f}x spread across cells",
        "nothing -- a FAIL falls back to the WIDEST cell, which is the "
        "conservative floor, and the published number says so"))
    return gates


def claim_gates(floors: dict[str, PooledFloor], spreads: list[CellSpread],
                control: dict[str, PairedDifference],
                cards: dict[str, PairedDifference],
                arms: list[Arm]) -> list[Gate]:
    """C-gates. A FAIL here is a result, not a broken run."""
    gates: list[Gate] = []
    primary = floors.get(PRIMARY_FIELD)
    sd = primary.pooled_sd if primary else None

    gates.append(Gate(
        "C1 floor size",
        f"the measured floor is no wider than the s3/s4 proxy implied "
        f"({PRIOR_SD:.4f})",
        f"pooled between-replicate sd of {PRIMARY_FIELD} <= {PRIOR_SD:.4f}",
        "PASS",
        None if sd is None else sd <= PRIOR_SD,
        "part (a) did not run" if sd is None
        else f"pooled sd {sd:.4f} on {primary.df} df, upper 95% "
             f"{primary.upper95:.4f}",
        "every N chosen against the prior: a wider floor means the whole study "
        "is scored against a looser instrument than it assumed"))

    delta_card = cards[PRIMARY_FIELD]
    gates.append(Gate(
        "C2 cross-card is under the floor",
        "the published cross-card difference is smaller than one run per card "
        "could resolve",
        "|paired mean| < the known-sigma MDE at the measured floor, n=1 per card",
        "PASS",
        None if sd is None or delta_card.mean is None
        else abs(delta_card.mean) < mde_external_sigma(sd, 1),
        "part (a) did not run" if sd is None or delta_card.mean is None
        else (f"|{delta_card.mean:+.4f}| against MDE "
              f"{mde_external_sigma(sd, 1):.4f} at the design the study ran, "
              f"one run per card"),
        "nothing -- a PASS here is the finding: the cross-card result is not a "
        "null, it is an unresolvable measurement"))

    resolvable = []
    for block_m in sorted({s.block_m for s in spreads}):
        got = two_sample_delta(spreads, arms[0].name, arms[-1].name, block_m)
        if got:
            delta, pooled_sd, n = got
            resolvable.append((block_m, delta, mde_two_sample(pooled_sd, n)))
    swizzle_ok = None
    if len(arms) >= 2 and arms[0].group_m != arms[-1].group_m and resolvable:
        swizzle_ok = all(abs(d) >= m for _, d, m in resolvable)
    gates.append(Gate(
        "C3 swizzle survives replicates",
        f"the GROUP_SIZE_M {arms[0].group_m} -> {arms[-1].group_m} swing is "
        f"bigger than the floor at every BLOCK_M",
        "|delta| >= the two-sample MDE at every measured BLOCK_M",
        "PASS",
        swizzle_ok,
        "the two arms do not differ in GROUP_SIZE_M, or nothing was measured"
        if swizzle_ok is None else
        "; ".join(f"BM={bm} {delta:+.4f} vs MDE {m:.4f}"
                  for bm, delta, m in resolvable),
        "the study's largest claim: if the swizzle swing is inside the floor "
        "then the alpha surface is noise"))

    ctrl = control[PRIMARY_FIELD]
    card = cards[PRIMARY_FIELD]
    separable = None
    if ctrl.mean is not None and card.mean is not None and ctrl.mde is not None:
        separable = abs(card.mean) - abs(ctrl.mean) >= ctrl.mde
    gates.append(Gate(
        "C4 card beats the stages control",
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
        "C5 sign consistency",
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


def git_state() -> dict:
    """Commit and dirtiness, so a published floor names the tree it came from."""
    out = {"commit": "", "dirty": None}
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=30)
        if rev.returncode == 0:
            out["commit"] = rev.stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        if status.returncode == 0:
            out["dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def paired_payload(diff: PairedDifference) -> dict:
    return {"first": diff.first, "second": diff.second,
            "sign": f"delta = alpha({diff.second}) - alpha({diff.first})",
            "n_cells": diff.n, "mean": diff.mean, "sd": diff.sd,
            "mde": diff.mde, "resolved": diff.resolved,
            "varied": list(diff.varied), "same_machine": diff.same_machine}


def build_document(control: dict[str, PairedDifference],
                   cards: dict[str, PairedDifference],
                   floors: dict[str, PooledFloor] | None,
                   *, n_replicates: int = 0, cache_mode: str = "",
                   gpu_name: str = "", provenance: str = "",
                   synthetic: bool = False) -> dict:
    """The published JSON. `replicate_floor` is null until a card produces one."""
    doc = {
        "schema": SCHEMA,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git_state(),
        "sign": "every delta is alpha(second arm) - alpha(first arm)",
        "primary_field": PRIMARY_FIELD,
        "effects_registered": [{"name": e.name, "size": e.size, "source": e.source}
                               for e in EFFECTS],
        "prior_sd": PRIOR_SD,
        "prior_sd_source": "s3-vs-s4 paired sd 0.0323 over 11 cells / sqrt(2); "
                           "CONFOUNDS num_stages, so an upper bound only",
        "stages_control": {name: paired_payload(d) for name, d in control.items()},
        "cross_card": {name: paired_payload(d) for name, d in cards.items()},
        "replicate_floor": None,
    }
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
                "gpu_name": gpu_name, "provenance": provenance,
                "synthetic": synthetic, "per_field": per_field,
            }
    return doc


def write_published(doc: dict, path: Path = NOISE_FLOOR_JSON) -> str:
    """Write the floor where git will take it, or refuse and say why."""
    accepted = git_accepts(path)
    if accepted is not True:
        why = ("git ignores it" if accepted is False
               else "git could not be asked whether it ignores it")
        raise SystemExit(
            f"REFUSING to write {path}: {why}. The rule is `results/*` ignored "
            f"with only `!results/published/` excepted; a floor written anywhere "
            f"else is a floor that disappears on commit.")
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
            "THE DESIGN THE STUDY ACTUALLY RAN is one run per condition with "
            "sigma imported from this",
            f"floor: {mde_external_sigma(sd, 1):.4f}. Everything below is what "
            f"REPLICATING would buy.",
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
        rows.append(f"    {effect.name:<34s}{effect.size:.4f}   "
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
        need = cells_for(EFFECTS[-1].size, card.sd)
        out.append(f"POWER:   at sd_d = {card.sd:.4f} the {card.n}-cell paired "
                   f"design detects {card.mde:.4f}. Detecting the observed "
                   f"{abs(card.mean):.4f} needs "
                   + ("more than 5000" if need is None else f"{need}")
                   + " matched cells, which is "
                   + ("" if need is None else
                      f"{need / card.n:.1f}x the cells actually run, i.e. "
                      f"{math.sqrt(need / card.n):.1f}x in standard error."))
    return "\n".join(out)


def render_plan(arms: list[Arm], n: int, cache_mode: str, base: Path,
                gpu_name: str, costs: dict[str, float | None]) -> str:
    out = ["## The plan", "",
           f"{len(arms)} arm(s) x {n} replicates = {len(arms) * n} sweep processes, "
           f"cache mode {cache_mode}, config device {gpu_name}",
           f"EVERYTHING IS SAVED TO  {base}", ""]
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
            run_id = run_id_for(arm, i, gpu_name=gpu_name, cache_mode=cache_mode)
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


def render_predictions(n: int) -> str:
    return f"""\
## Predictions, registered before anything ran

VALIDITY -- a FAIL means no number from part (a) may be quoted.
  V1  every replicate got its own run id and directory   distinct ids == {n} x arms
  V2  the planned replicates ran and produced fits       >= 2 cells at n = {n}
  V3  no cell returned identical alpha every time        zero degenerate cells
  V4  the Triton cache behaved as the mode claims        fresh: >= 1 artefact each
  V5  the cells share a spread                           max sd / min sd <= {HOMOGENEITY_RATIO:g}

CLAIM -- a FAIL is a result, not a broken run. Expected verdict in brackets.
  C1  the floor is no wider than the proxy implied  [PASS]  sd <= {PRIOR_SD:.4f}
  C2  the cross-card difference is under the floor  [PASS]  |{EFFECTS[-1].size:.4f}| < MDE
  C3  the swizzle swing survives replicates         [PASS]  |delta| >= MDE at every BM
  C4  the card beats the num_stages control         [FAIL]  |card| - |stages| >= MDE
  C5  the cross-card sign is estimator-independent  [FAIL]  one sign across three fields

C4 and C5 are registered as EXPECTED FAILURES from the committed reports alone, and
both are computed here without a GPU. C4 fails because the same 11 cells move
+0.0101 when num_stages goes 4 -> 3 on one card and +0.0117 when the card changes.
C5 fails because the cross-card difference is +0.0117 in alpha-corrected and
-0.3325 in alpha-upper -- two anchorings of the same fit, opposite signs -- while
the num_stages control keeps one sign across all three. Registering them as
failures now means neither can be reported later as a discovery."""


# --------------------------------------------------------------------------
# running the replicates
# --------------------------------------------------------------------------

DRY_COST_RE = re.compile(r"estimated GPU time\s+([0-9.]+)\s*s")


def sweep_cost(arm: Arm, python: str) -> float | None:
    """Ask the sweep itself what one arm costs. None when it will not say.

    None rather than a guess: a fabricated cost on a metered pod is how a
    session runs out of budget three arms from the end.
    """
    argv = [python, str(SWEEP), "--dry-run"] + arm.sweep_argv("cost-probe", Path("/tmp"))
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
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
                  shared_cache: Path | None, timeout_s: float) -> Replicate:
    """One sweep process. Its log survives even when it fails."""
    run_id = run_id_for(arm, index, gpu_name=gpu_name, cache_mode=cache_mode)
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
        if done.returncode != 0:
            rep.error = (f"sweep exited {done.returncode}; see {log}")
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
    ap.add_argument("--arms", default="mixtral_g1,mixtral_g16",
                    help="comma list from " + ",".join(a.name for a in DEFAULT_ARMS))
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


def main(argv: list[str] | None = None) -> int:
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
    print(render_power_table(PRIOR_SD, "the s3/s4 proxy, an UPPER bound",
                             cells=CELLS_PER_ARM * len(arms)))

    if args.control_only:
        print()
        print("=" * 72)
        print("PART (b) ONLY. The replicate floor was not measured and no number "
              "on this page is one.")
        print("=" * 72)
        if args.publish:
            print(write_published(build_document(control, cards, None)))
        print()
        print(IMPORT_BANNER)
        return EXIT_NOT_MEASURED

    device, missing = detect_gpu()
    gpu_name = args.gpu_name or device or "NVIDIA H200"
    base = (args.out_dir or default_out_base()) / f"{cache_mode}-n{n}"
    costs = {arm.name: sweep_cost(arm, args.python) for arm in arms}

    print()
    print(render_plan(arms, n, cache_mode, base, gpu_name, costs))
    print()
    print(render_predictions(n))

    rehearsing = args.rehearse is not None
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
        print("=" * 72)
        if args.publish:
            print(write_published(build_document(control, cards, None)))
        print()
        print(IMPORT_BANNER)
        return EXIT_NOT_MEASURED

    extra: list[str] = list(args.sweep_arg)
    shared_cache = (base / "shared-triton-cache") if args.warm_cache else None
    if rehearsing:
        extra += ["--self-test", str(args.rehearse),
                  "--self-test-noise", str(args.rehearse_noise)]
        print()
        print("REHEARSAL: every replicate below runs the sweep's --self-test, so "
              "its cells are GENERATED from the model and nothing is measured. "
              "The seed varies per replicate purely to make the plumbing move; a "
              "real replicate holds the seed FIXED. Stamped synthetic everywhere.")
        print("  C1 and C3 are MEANINGLESS here and their verdicts must be "
              "ignored: the floor is whatever --rehearse-noise was set to, and "
              "the sweep's synthetic cells do not depend on GROUP_SIZE_M, so the "
              "two arms are the same data and C3 reads a swizzle delta of exactly "
              "0.0000. That C3 catches it is the point of running this.")

    base.mkdir(parents=True, exist_ok=True)
    replicates: list[Replicate] = []
    started = time.time()
    for arm in arms:
        for index in range(1, n + 1):
            per_rep = list(extra)
            if rehearsing:
                # The sweep's synthetic cells are a deterministic function of the
                # seed, so a rehearsal that held it fixed would produce N
                # identical reports, sd 0.0000, and would exercise nothing except
                # the V3 gate. A real replicate does the opposite: seed FIXED.
                per_rep += ["--seed", str(1000 + index)]
            print(f"  [{arm.name} rep {index}/{n}] launching", flush=True)
            rep = run_replicate(arm, index, base, gpu_name=gpu_name,
                                cache_mode=cache_mode, python=args.python,
                                extra=per_rep, shared_cache=shared_cache,
                                timeout_s=args.replicate_timeout)
            replicates.append(rep)
            status = "ok" if rep.ok else f"FAILED: {rep.error}"
            print(f"      {rep.seconds:.0f} s  {status}", flush=True)

    floors = {f: pool(spreads_for(replicates, f), f) for f in ALPHA_FIELDS}
    primary_spreads = spreads_for(replicates, PRIMARY_FIELD)

    gates = validity_gates(replicates, n, arms, cache_mode, floors)
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
            print(f"  {name:16s} floor sd {floor.pooled_sd:.4f} on {floor.df} df, "
                  f"upper 95% {floor.upper95:.4f} -- {floor.reason}")

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
                  f"{cache_mode}, {time.time() - started:.0f} s wall, under {base}")
    publishable = cache_mode == args.floor_from
    doc = build_document(
        control, cards, floors if publishable else None,
        n_replicates=n, cache_mode=cache_mode, gpu_name=gpu_name,
        provenance=provenance, synthetic=rehearsing)
    (base / "noise_floor.json").write_text(json.dumps(doc, indent=2) + "\n")
    print()
    print(f"EVERYTHING IS SAVED TO {base}")
    print(f"  summary {base / 'noise_floor.json'}")
    if not publishable:
        print(f"  the {cache_mode} floor is NOT publishable as THE floor "
              f"(--floor-from is {args.floor_from}); replicate_floor is null above")
    if args.publish and rehearsing:
        # A synthetic floor in the curated directory would be readable by anyone
        # who passed allow_synthetic, and would sit in git looking exactly like a
        # measurement. --rehearse and --publish are mutually exclusive on purpose.
        print(f"  REFUSING --publish: this was a REHEARSAL. {NOISE_FLOOR_JSON} "
              f"only ever holds numbers measured on a card. The synthetic summary "
              f"is under {base} and nothing else was written.")
    elif args.publish:
        print(write_published(doc))
    else:
        print(f"  --publish would write {NOISE_FLOOR_JSON}")

    print()
    print(IMPORT_BANNER)

    failed = [g for g in gates if g.passed is False]
    return EXIT_GATE_FAILED if failed else EXIT_OK


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
