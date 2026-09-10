"""Roofline analysis over a results CSV.

Reads peaks from a cited hardware file and refuses to draw an uncited roof.
Arithmetic intensity comes from bytes_model, which computes it per tiling, so a
fused pipeline sits at a different x position than an unfused one and the
predicted benefit of a fusion is readable off the plot before it is measured.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import schema as SC

HARDWARE_DIR = Path(__file__).parent / "hardware"


class HardwareMismatch(RuntimeError):
    """A calibration file describes a different GPU than the one attached."""


class UnverifiedHardware(RuntimeError):
    pass


@dataclass(frozen=True)
class Hardware:
    name: str
    bandwidth_bytes_s: float
    peak_flops: dict[str, float]
    source: str
    #: Board power limit in watts. The only cheap way to tell an H200 SXM
    #: (700 W) from an H200 NVL (600 W): torch reports both as "NVIDIA H200".
    tdp_w: float | None = None
    #: For a measured profile, which STREAM pattern defined the bandwidth.
    #: Empty for a datasheet profile, where the figure is a pin rate.
    ceiling_pattern: str = ""
    #: The clock each compute roof was measured at, keyed by REFERENCE FAMILY
    #: (`reference_family`: "bf16" for the dense bf16 GEMM every non-fp8 dtype
    #: is scored against, "fp8" for the fp8 GEMM), read out of the SAME file
    #: as the peaks by `load_hardware`. Empty for a datasheet profile, which
    #: was never measured at any clock. Carried here so a per-row roof can be
    #: re-derived off-GPU from the one file the roof came from: `recompute`
    #: used to rebuild the fixed-roof columns from a `Hardware` that had
    #: dropped the clock, and left `roof_at_cell_clock_tflops` at the old
    #: peak's value beside a rewritten `achieved_peak_tflops`.
    reference_clocks: dict[str, ReferenceClock] = field(default_factory=dict)

    def peak(self, dtype: str) -> float:
        v = self.peak_flops.get(dtype)
        if not v:
            raise ValueError(
                f"{self.name}: no verified peak for dtype {dtype!r}; "
                f"fill it in {HARDWARE_DIR}"
            )
        return v

    def ridge_point(self, dtype: str) -> float:
        """FLOP per byte above which a kernel can be compute bound."""
        return self.peak(dtype) / self.bandwidth_bytes_s

    def attainable(self, dtype: str, arithmetic_intensity: float) -> float:
        return min(self.peak(dtype), arithmetic_intensity * self.bandwidth_bytes_s)

    def bound(self, dtype: str, arithmetic_intensity: float) -> str:
        return "compute" if arithmetic_intensity >= self.ridge_point(dtype) else "memory"


def load_hardware(name: str = "h200_nvl", allow_unverified: bool = False,
                  directory: Path | None = None) -> Hardware:
    import yaml

    path = (directory or HARDWARE_DIR) / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no hardware file {path}")
    data = yaml.safe_load(path.read_text())

    if not data.get("verified") and not allow_unverified:
        raise UnverifiedHardware(
            f"{path} is marked verified: false. Check the peaks against "
            f"{data.get('source')} (use DENSE, not sparsity figures), set "
            "checked_by/checked_on, then flip verified to true. Passing "
            "allow_unverified=True is for exploration only and taints any plot."
        )

    bw = data["memory"]["bandwidth_tb_s"]
    if not bw:
        raise ValueError(f"{path}: memory.bandwidth_tb_s is null")

    peaks = {k: (v * 1e12 if v else None)
             for k, v in (data.get("compute_dense_tflops") or {}).items()}
    card = str((data.get("detail") or {}).get("gpu_name") or "")
    return Hardware(
        name=data["name"],
        bandwidth_bytes_s=float(bw) * 1e12,
        peak_flops={k: v for k, v in peaks.items() if v},
        source=data.get("source", ""),
        ceiling_pattern=(data.get("detail") or {}).get("ceiling_pattern", ""),
        tdp_w=data.get("tdp_w"),
        # Only the families this file records a clock for. A datasheet
        # profile has no `detail` and gets none; a measured profile whose
        # fp8 GEMM was refused (the A100 writes fp8_gemm_clock_mhz: 0) gets
        # the bf16 one alone, and an fp8 row scored against it is refused by
        # name rather than levelled against the other GEMM's clock.
        reference_clocks={
            family: ref for family in REFERENCE_FAMILIES
            if (ref := reference_clock_from_doc(data, card, family)).mhz},
    )


def peak_bandwidth(name: str = "h200_nvl") -> float | None:
    """Bandwidth for cost prediction. Returns None if unavailable rather than
    guessing, and callers fall back to doing the measurement."""
    try:
        return load_hardware(name, allow_unverified=True).bandwidth_bytes_s
    except (FileNotFoundError, ValueError, KeyError):
        return None


def device_name_matches(a: str, b: str) -> bool:
    """Do two card names name the same part? Loose on purpose, and one rule.

    Datasheet names ("NVIDIA H200 NVL"), torch's device names ("NVIDIA H200")
    and a calibration's own `name` ("NVIDIA H200 (measured)") agree on the part
    and not on spacing or suffixes, so the comparison is substring-after-
    normalising in either direction. Written once and called by everything in
    this module that compares two card names, because the rule has already been
    written twice in this repository and the two copies differed on non-ASCII
    alphanumerics.

    An empty name on either side is not a match. `device_matches` keeps its own
    "nothing to check against" allowance, which is a different question.
    """
    def norm(text: str) -> str:
        return "".join(c for c in str(text).lower() if c.isalnum())

    x, y = norm(a), norm(b)
    return bool(x) and bool(y) and (x in y or y in x)


def device_matches(hw: Hardware, gpu_name: str) -> bool:
    """Does this hardware profile describe the GPU the rows were measured on?

    Plotting an H100 run against an H200 roof would understate efficiency by
    the ratio of their peaks and is exactly the kind of error this repo exists
    not to make. An empty `gpu_name` is nothing to check against and passes.
    """
    if not gpu_name:
        return True                      # nothing to check against
    return device_name_matches(hw.name, gpu_name)


def available_profiles() -> list[str]:
    return sorted(p.stem for p in HARDWARE_DIR.glob("*.yaml"))


def power_limit_w() -> float | None:
    """Board power limit, or None. Used to disambiguate same-named parts."""
    from .timing import _nvidia_smi

    vals = _nvidia_smi("power.limit")
    if not vals:
        return None
    try:
        return float(vals[0].split()[0])
    except (ValueError, IndexError):
        return None


def for_device(gpu_name: str, tdp_w: float | None = None) -> str | None:
    """Pick the hardware profile describing this GPU, or None if it is unclear.

    AMBIGUITY MUST NOT RESOLVE SILENTLY. torch reports an H200 SXM as plain
    "NVIDIA H200", which is a substring of both "NVIDIA H200 NVL" and
    "NVIDIA H200 SXM". Returning the first match picked NVL, whose BF16 peak is
    835.5 against SXM's 989.5, and every efficiency number would have been
    understated by 18% with nothing to indicate it.

    So: exactly one match is an answer; several is a question. A missing roof is
    recoverable and loud; a wrong one is neither.
    """
    matched = []
    for stem in available_profiles():
        if is_measured_profile(stem):
            continue
        try:
            hw = load_hardware(stem, allow_unverified=True)
        except (ValueError, KeyError):
            continue
        if device_matches(hw, gpu_name):
            matched.append((stem, hw))

    # An ambiguous name is resolvable by power limit: an H200 SXM is 700 W and
    # an H200 NVL is 600 W, and torch calls both "NVIDIA H200".
    if len(matched) > 1 and tdp_w is not None:
        by_power = [stem for stem, hw in matched
                    if hw.tdp_w is not None and abs(hw.tdp_w - tdp_w) < 25]
        if len(by_power) == 1:
            return by_power[0]

    return matched[0][0] if len(matched) == 1 else None


def ambiguous_for_device(gpu_name: str) -> list[str]:
    """Profiles that all match this device name. Non-empty means unresolvable."""
    matched = []
    for stem in available_profiles():
        if is_measured_profile(stem):
            continue
        try:
            hw = load_hardware(stem, allow_unverified=True)
        except (ValueError, KeyError):
            continue
        if device_matches(hw, gpu_name):
            matched.append(stem)
    return matched if len(matched) > 1 else []


#: What `load_measured` appends to the device name in `Hardware.name`.
MEASURED_DECORATION = " (measured)"


def measured_slug(gpu_name: str) -> str:
    """Filename stem for this device's calibration.

    One harness, one calibration per device. A single shared `measured.yaml`
    meant calibrating on a second GPU overwrote the first, and a later re-plot
    of the published sweep then scored it against the wrong roof.
    """
    # `load_measured` hands back `Hardware.name` as "<device> (measured)".
    # On 2026-09-09 bm128_depth passed that name here, got the stem
    # measured_nvidia_h200_measured, found no file, and REFUSED a sound
    # calibration before timing a tread. The decoration is stripped so the
    # slug is the device's whichever name a caller has in hand.
    if gpu_name.endswith(MEASURED_DECORATION):
        gpu_name = gpu_name[:-len(MEASURED_DECORATION)]
    safe = "".join(c if c.isalnum() else "_" for c in gpu_name.lower())
    while "__" in safe:
        safe = safe.replace("__", "_")
    return f"measured_{safe.strip('_')}"


def is_measured_profile(stem: str) -> bool:
    """Is this yaml a calibration of a machine rather than a datasheet?

    `for_device` searches DATASHEET profiles. A measured file is this box's own
    calibration, and its name is "<device> (measured)", which contains the
    device name as a substring. Letting one into that search makes a device
    ambiguous with itself the moment it is calibrated.
    """
    return stem == "measured" or stem.startswith("measured_")


def current_gpu_name() -> str:
    """torch's name for the attached GPU, or "" when there is no CUDA device."""
    try:
        import torch
        if not torch.cuda.is_available():
            return ""
        return torch.cuda.get_device_properties(0).name
    except Exception:  # noqa: BLE001  - absent torch, driver error, no device
        return ""


def load_measured(gpu_name: str | None = None,
                  directory: Path | None = None) -> Hardware | None:
    """Ceilings measured by scripts/calibrate_hardware.py for THIS device.

    calibrate_hardware.py writes its yaml in exactly load_hardware's schema, so
    this is a lookup rather than a second parser. Returns None when no
    calibration has been run, so callers leave the efficiency columns empty
    instead of quoting a datasheet peak.

    Prefers the per-device file. Falls back to a bare `measured.yaml` only when
    it describes this device, and raises otherwise: scoring an A100 run against
    a committed H200 calibration produces rows that look entirely plausible and
    are wrong by the ratio of two machines' ceilings.
    """
    if gpu_name is None:
        gpu_name = current_gpu_name()

    names = [measured_slug(gpu_name)] if gpu_name else []
    names.append("measured")
    for name in names:
        try:
            hw = load_hardware(name, directory=directory)
        except (FileNotFoundError, ValueError, KeyError, UnverifiedHardware):
            continue
        if device_matches(hw, gpu_name):
            return hw
        raise HardwareMismatch(
            f"{name}.yaml was measured on {hw.name!r} but this machine reports "
            f"{gpu_name!r}. Those ceilings are not this machine's, and every "
            "efficiency column derived from them would be wrong by the ratio "
            "of the two parts. Run:\n"
            f"    python scripts/calibrate_hardware.py\n"
            f"which writes {measured_slug(gpu_name)}.yaml for this device.")
    return None


#: The GRADES a reference clock can carry, i.e. how the number was taken.
#: Only `REFERENCE_UNDER_LOAD` is a sample of the clock the roof's GEMM was
#: actually running at. The other two are disowned for scaling the roof per
#: row (`roof_at_clock`), and a LEVEL verdict scored against them is
#: provisional: the row carries the grade so a reader can discount it.
REFERENCE_UNDER_LOAD = "under-load"
REFERENCE_IDLE_SCALAR = "idle-scalar"
REFERENCE_SETTLE_PLATEAU = "settle-plateau"

#: Where a calibration records the clock its dense GEMM ran at, in falling
#: order of directness, with what each field is and the grade it earns.
#: `calibrate.clock_established` exists because the three have disagreed:
#: eleven committed calibrations of one H200 recorded 1485-1935 MHz for the
#: scalar while their own settle histories sat at 1455-1515. The order prefers
#: the number measured UNDER THE LOAD THAT SET THE ROOF, and the record says
#: which field was read, so a LEVEL exclusion can always be traced back to a
#: field in a file.
#:
#: THE SECOND FIELD IS DISOWNED AND STILL READ. `detail.gemm_clock_mhz` is the
#: post-hoc idle sample `calibrate.py`'s own header records a 30% spread for
#: (1485-1935 MHz across eleven calibrations of one card), and both committed
#: calibrations carry ONLY that field: they predate the `gemm_clock` block. It
#: is read rather than refused because a run against those files must still
#: be able to say something about LEVEL, and what it says is graded: the
#: source names the artefact, the row carries `reference_clock_source`, and
#: `ReferenceClock.usable_for_roof` is False, so the per-row roof is not
#: rescaled against a number that could be 30% off. A recalibration writes
#: the under-load median and the grade rises with it. The order is NOT
#: changed to put the settle plateau ahead of the scalar, although the
#: plateau is the steadier number: `scripts/block_m_crossing_sweep.py`
#: carries its own copy of this walk, `tests/test_driver.py` pins the two
#: resolvers to one number per card, and a reorder here alone would level the
#: driver and the ladders against different references.
#: `scripts/dtype_tile_confound.py:_reference_clock` is a THIRD copy, and
#: since the reference went per family it is a divergent one: it walks the
#: bf16 fields only and hands that clock to the fp8 cells the arm exists to
#: time, so on the committed H200 an fp8 cell at the fp8 GEMM's own 1395 MHz
#: is level under the driver and LEVEL-failed LOW under that script (1395
#: against the bf16 GEMM's 1485 is 0.939, under the band's snapped 1410 edge).
#: Before ab61e55 the same divergence ran the other way, HIGH, on the idle
#: scalars that file then carried: 1905 against 1515. A
#: script with a private walk should call `reference_clock_from_doc(raw,
#: name, reference_family(dtype))` per cell; `tests/test_roofline.py` pins
#: that call's contract on the committed file so the replacement is a swap.
#: THE REFERENCE IS PER GEMM, AND A CALIBRATION RUNS TWO. `calibrate.py`
#: measures a dense bf16 GEMM and, on silicon that has the format, a dense fp8
#: GEMM, and records the clock each ran at: the committed H200 file carries
#: `gemm_clock_mhz: 1485` beside `fp8_gemm_clock_mhz: 1395`, the under-load
#: medians of ab61e55. Those are two roofs at two clocks, and a row's
#: reference is the clock of the roof ITS dtype is scored against. Until
#: 2026-09-08 every row was levelled against the bf16 number, and against the
#: idle scalars that file carried then (1515 and 1905) an fp8 cell running at
#: the fp8 roof's own 1905 MHz was filed LEVEL-failed HIGH, while
#: `roof_at_clock(1447.7, 1515, 1905)` would have rescaled the fp8 roof to
#: 1820 TFLOP/s, 26% above the figure the calibration measured at that very
#: clock. Two families, named by the GEMM: "bf16" is the
#: dense bf16 GEMM, the roof for bf16 and for fp16 (the yaml publishes one
#: number for both) and the only compute reference anything else has; "fp8"
#: is the fp8 GEMM, for the `spec.FP8_WEIGHT_DTYPES`.
BF16_FAMILY = "bf16"
FP8_FAMILY = "fp8"
REFERENCE_FAMILIES = (BF16_FAMILY, FP8_FAMILY)


def reference_family(dtype: str) -> str:
    """Which GEMM's clock a row of `dtype` is levelled against. Pure, total.

    Every fp8 weight format is scored against the fp8 GEMM's roof, so it is
    levelled against that GEMM's clock; everything else against the bf16
    GEMM's. Total rather than refusing on an unknown dtype because the roof
    lookup (`Hardware.peak`) is where an unknown dtype is refused, with the
    file to fill in, and a second refusal here would fire first with less to
    say. The two calls agree on what "fp8" means: `spec.FP8_WEIGHT_DTYPES`.
    """
    from ..spec import FP8_WEIGHT_DTYPES
    return FP8_FAMILY if dtype in FP8_WEIGHT_DTYPES else BF16_FAMILY


CLOCK_FIELDS = (
    (("detail", "gemm_clock", "median_mhz"),
     "the median of the samples taken while the calibration's dense GEMM ran",
     REFERENCE_UNDER_LOAD),
    (("detail", "gemm_clock_mhz"),
     "gemm_clock_mhz, the scalar the calibration published for its dense GEMM; "
     "DISOWNED: a single post-hoc sample taken after the GEMM had synchronised, "
     "with the GPU idle and boosting, which calibrate.py records moving 30% "
     "across eleven calibrations of one card. LEVEL against it is provisional "
     "and the per-row roof is not rescaled against it; recalibrate to record "
     "the under-load median",
     REFERENCE_IDLE_SCALAR),
    (("detail", "settle", "final_mhz"),
     "the compute settle's final plateau; this calibration recorded no clock "
     "sampled during the GEMM itself. DISOWNED for the per-row roof: the "
     "plateau is a different kernel's steady state, within 10% of the GEMM's "
     "clock by calibrate.CLOCK_VS_SETTLE_TOL_PCT, which is a bracket",
     REFERENCE_SETTLE_PLATEAU),
)

#: The fp8 GEMM's clock, in the same falling order and with the same grades.
#: NO SETTLE-PLATEAU FALLBACK, deliberately: the settle is run under the bf16
#: GEMM's power regime, and the committed H200 file puts the fp8 GEMM 435 MHz
#: above that plateau (1905 against 1470), far outside the 10% bracket that
#: makes the plateau a stand-in for the bf16 GEMM. A file with neither fp8
#: field has no fp8 reference, and says so.
FP8_CLOCK_FIELDS = (
    (("detail", "fp8_gemm_clock", "median_mhz"),
     "the median of the samples taken while the calibration's dense fp8 GEMM "
     "ran", REFERENCE_UNDER_LOAD),
    (("detail", "fp8_gemm_clock_mhz"),
     "fp8_gemm_clock_mhz, the scalar the calibration published for its dense "
     "fp8 GEMM; DISOWNED for the same reason as gemm_clock_mhz: a single "
     "post-hoc sample taken with the GPU idle and boosting. LEVEL against it "
     "is provisional and the per-row roof is not rescaled against it; "
     "recalibrate to record the under-load median",
     REFERENCE_IDLE_SCALAR),
)

CLOCK_FIELDS_BY_FAMILY = {BF16_FAMILY: CLOCK_FIELDS, FP8_FAMILY: FP8_CLOCK_FIELDS}


def roof_at_clock(peak_tflops: float | None, reference_mhz: float | None,
                  load_mhz: float | None) -> float | None:
    """The compute roof AT THE CLOCK A CELL RAN, from the roof at its reference.

    `peak_tflops` is the calibration's achieved dense-GEMM figure, measured at
    `reference_mhz` (the under-load median of that GEMM's own clock). The
    tensor-core issue rate is linear in the SM clock, so the same GEMM at
    `load_mhz` would reach `peak * load / reference`; that is the roof a cell
    which ran at `load_mhz` should be scored against, and it is what
    `calibrate.sustained_peak_tflops(load_mhz)` gives for the silicon,
    multiplied by the cuBLAS efficiency the calibration measured. The ratio
    form is used here so it works off-GPU from a CSV, with the one assumption
    stated: cuBLAS's fraction of the silicon peak is taken as clock-invariant
    across the band a cell can sit in.

    WHY THIS EXISTS. A memory-shaped decode cell on an H200 runs at ~1980 MHz;
    the compute roof was measured at ~1515, power-limited. Scored against the
    fixed roof the cell's fraction is inflated by 1980/1515 = 1.31x, toward
    the study's claim, and the one-sided LEVEL flag passed it. The bandwidth
    roof is NOT rescaled: HBM does not run on the SM clock (calibrate.py
    measured 1.7% sensitivity) and the ridge moves with this number alone.

    None, never a substitute, when any term is missing or non-positive: no
    load clock on the row (the retired seam, an NVML-less container), no
    reference, or a reference the caller has not graded as under-load. The
    caller decides whether to score at all; this refuses to invent a roof.
    """
    if not peak_tflops or not reference_mhz or not load_mhz:
        return None
    if peak_tflops <= 0 or reference_mhz <= 0 or load_mhz <= 0:
        return None
    return float(peak_tflops) * float(load_mhz) / float(reference_mhz)


def roof_scale_refusal(load_mhz: float | None, reference_mhz: float | None,
                       grade: str) -> str:
    """Why the per-row roof cannot be scored for a row, or "" when it can. Pure.

    THE ONE STATEMENT OF THE RULE, read by both writers of the per-row roof:
    `driver._apply_cost` on the pod and `recompute.ceiling_columns` off it.
    The recompute used to carry no statement at all and left the column at the
    old peak's value, which is the shape this repository keeps producing (a
    rule applied at one of two call sites), so the rule now lives where both
    sites have to import it.

    Three refusals, each named so `roof_note` says which: no under-load clock
    on the row (the retired seam writes none; an NVML-less container polls
    none), no reference resolved for the row's dtype family, or a reference
    whose grade is not the under-load median. The third fired against BOTH
    committed calibrations until 2026-09-09; since ab61e55 the H200 file
    records the under-load medians its own pod measured (bf16 1485 MHz, fp8
    1395) and its rows score, and the A100 file alone still carries the
    post-hoc idle scalar (1335 MHz) `calibrate.py` records a 30% spread for.
    Scaling a roof by `load / reference` against that scalar would move every
    fraction by up to that much under the name of a correction. REFUSED rather
    than defaulted to the fixed roof: the fixed-roof figure is still on the
    row under its own name, with its bias stated, and this column stays 0.0
    with the reason.
    """
    if not load_mhz or load_mhz <= 0:
        return ("no under-load clock on this row (retired seam, or no NVML "
                "during the trials), so the roof cannot be placed at the "
                "clock the cell ran")
    if reference_mhz is None or reference_mhz <= 0:
        return ("no reference clock was resolved for this row's dtype family, "
                "so the roof has no clock to be rescaled from")
    if grade != REFERENCE_UNDER_LOAD:
        return (f"the reference is graded {grade or 'none'!r}, "
                "not an under-load median; rescaling the roof against it "
                "would put a disowned number into every fraction (see "
                "reference_clock_source); recalibrate to record the "
                "under-load median")
    return ""


#: What `roof_note` says on a row that IS scored: which of the two compute
#: fractions a gate reads, and what the other one is for.
#:
#: WHY THE FIXED ROOF IS THE GATE INPUT. Both fractions are on every row and
#: they differ by exactly `load / reference`. The calibration's GEMM and every
#: cell of a session run under the SAME board power cap, so the fixed roof
#: compares delivered throughput under one budget, which is the comparison a
#: compute-bound claim is about; the own-clock fraction divides that by the
#: clock the cell happened to hold, which credits a tile for its own power
#: draw. On the 2026-09-09 H200 session that difference flips the sign of the
#: roofline arm's control-subject gap (+0.053 fixed, -0.032 own clock), so it
#: is not a presentational choice. Memory-shaped cells and cross-card work
#: read the own-clock fraction, labelled as issue efficiency.
ROOF_NOTE_SCORED = (
    "scored: pct_of_achieved_tflops (fixed roof) is the compute-bound gate "
    "input, since every cell and the calibration GEMM ran under one power cap; "
    "pct_of_roof_at_cell_clock is issue efficiency at this row's own clock, "
    "printed beside it; it is not the compute-bound gate input, and "
    "memory-shaped and cross-card work read it labelled as issue efficiency")


def cell_clock_roof(peak_tflops: float, load_mhz: float | None,
                    reference_mhz: float | None, grade: str
                    ) -> tuple[float, str]:
    """`(roof_at_cell_clock_tflops, roof_note)` for one row. Pure.

    The pair the two writers put on a row, computed once: `(roof,
    ROOF_NOTE_SCORED)` when `roof_scale_refusal` is empty, `(0.0, why)`
    otherwise. 0.0 is "not scored", never a roof of zero;
    `schema.has_cell_clock_roof` reads it so, and never reads the note.

    THE NOTE IS NOT EMPTY ON A SCORED ROW, and that is the 2026-09-09 change.
    Both fractions were already written; nothing on the row said which one a
    gate reads, so each of six consumers decided for itself and two of them
    disagreed with the arm's own text. The rule now travels with the number.
    """
    why = roof_scale_refusal(load_mhz, reference_mhz, grade)
    if why:
        return 0.0, why
    roof = roof_at_clock(peak_tflops, reference_mhz, load_mhz)
    # `roof_at_clock` refuses exactly the inputs the refusal above named, and
    # `peak_tflops` is the caller's positive roof; a None here would be a
    # contract change between the two, not a row.
    assert roof is not None
    return roof, ROOF_NOTE_SCORED


@dataclass(frozen=True)
class ReferenceClock:
    """The SM clock this card's roof was measured at, and where it came from.

    `timing.clock_flags` cannot say anything about LEVEL without one: a level is
    relative to something, and `time_kernel` will not invent the something. The
    something is the clock the CALIBRATION ran its GEMM at, because the roof
    every cell is scored against is that GEMM's number.

    `source` is ONE string doing two jobs: where the number came from when
    `mhz` is set, and why it is not known when `mhz` is None. One field rather
    than two so a provenance and a reason cannot drift apart, and so a caller
    that prints `source` prints something true either way.

    `grade` is HOW the number was taken, one of the `REFERENCE_*` constants or
    "" when there is no number. It is the field a consumer branches on:
    `usable_for_roof` is True only for the under-load median, because scaling
    a roof by `load / reference` puts the reference INTO the number, and a
    reference the calibration itself disowns (the idle scalar, 30% spread)
    would move every fraction-of-roof by up to that much under the name of a
    correction. The LEVEL flag is still scored against a disowned reference,
    with the grade written into every row so the verdict can be discounted.

    `card` is the attached device, and its emptiness is the load-bearing
    distinction for a caller deciding what to do about a None `mhz`: a card
    with no usable calibration is a POD MISCONFIGURATION, and no card at all is
    a laptop, where nothing is being measured against a clock in the first
    place. `profile` is the `name:` of the calibration the number was read out
    of, so a caller can check that the clock and the roof came from ONE file.
    """

    mhz: float | None
    source: str
    card: str = ""
    profile: str = ""
    grade: str = ""
    #: Which GEMM's clock this is, one of `REFERENCE_FAMILIES`. The bf16 GEMM
    #: unless a caller asked for the other, because that is the roof every
    #: non-fp8 dtype is scored against and the only clock a calibration is
    #: guaranteed to have measured; `reference_family` maps a dtype to it.
    family: str = BF16_FAMILY

    @property
    def usable_for_roof(self) -> bool:
        """May the roof be rescaled per row against this reference?"""
        return self.mhz is not None and self.mhz > 0 and self.grade == REFERENCE_UNDER_LOAD


def _dig(doc: dict, path: tuple[str, ...]):
    """`doc["a"]["b"]` for a path, or None if any level is missing or not a
    dict. A calibration written by an older `calibrate_hardware.py` is missing
    whole blocks, and a KeyError here would be a crash where the answer is
    "that file does not record it"."""
    node = doc
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def measured_doc(gpu_name: str | None = None, directory: Path | None = None
                 ) -> tuple[dict, str]:
    """The calibration yaml `load_measured` would use, parsed, plus the reason
    when there is none.

    Read as a document rather than through `Hardware`, which carries only the
    headline peaks and drops everything under `detail` -- the clocks included.
    That is the whole point of the function: the roof and the clock the roof
    was measured at are two readings of ONE calibration, and resolving them
    separately is how a run ends up levelled against one card while scored
    against another.

    SO THE CANDIDATE TEST IS `load_hardware` ITSELF, not a restatement of it.
    This function walked `load_measured`'s candidate NAMES in `load_measured`'s
    order and then accepted the first file that merely existed and parsed,
    while `load_measured` goes through `load_hardware`, which SKIPS a candidate
    on `UnverifiedHardware`, on a null bandwidth and on a missing block. Those
    two rules diverge on exactly the file that matters: a `verified: false`
    `measured_<card>.yaml` beside a valid `measured.yaml` gave the roof from
    one file and the clock from the other, and the caller-side cross-check
    could not see it because both carry the same `name:`. Demonstrated at
    1935 MHz against a 1470 MHz roof. Asking `load_hardware` costs a second
    parse of one small file, once per run, and it cannot drift.

    The device check is the same rule, `device_name_matches`, over a stricter
    field: `detail.gpu_name`, which is what torch reported on the calibrating
    box, falling back to the `name:` that `load_measured` compares. They agree
    on both committed calibrations ("NVIDIA H200" against "NVIDIA H200
    (measured)"), and where they could not, the raw device string is the one
    that answers "was this file written on this card".

    Does not raise where `load_measured` does. A file that describes another
    machine is reported as a reason, because this is a "record what you can and
    name what you cannot" reader; `load_measured` raising `HardwareMismatch`
    first is what actually stops such a run.
    """
    import yaml

    if gpu_name is None:
        gpu_name = current_gpu_name()
    if not gpu_name:
        return {}, ("no CUDA device is attached, so there is no card whose "
                    "calibration could say what clock the roof was measured at")
    looked: list[str] = []
    skipped: list[str] = []
    for stem in (measured_slug(gpu_name), "measured"):
        path = (directory or HARDWARE_DIR) / f"{stem}.yaml"
        looked.append(path.name)
        if not path.exists():
            continue
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except Exception as exc:                          # noqa: BLE001
            return {}, f"{path.name} did not parse: {type(exc).__name__}: {exc}"
        try:
            load_hardware(stem, directory=directory)
        except (FileNotFoundError, ValueError, KeyError,
                UnverifiedHardware) as exc:
            skipped.append(f"{path.name} ({type(exc).__name__})")
            continue
        named = str(_dig(data, ("detail", "gpu_name")) or data.get("name") or "")
        if named and not device_name_matches(named, gpu_name):
            return {}, (f"{path.name} was measured on {named!r} and this "
                        f"machine reports {gpu_name!r}; those are not the same "
                        "card's numbers")
        return data, ""
    if skipped:
        return {}, (f"no usable calibration for {gpu_name!r}: "
                    f"{', '.join(skipped)} exists, but `load_measured` skips "
                    "it for that reason and the roof did not come from it "
                    "either, so a clock read out of it would be levelling "
                    "against a file nothing else in this run used. Fix the "
                    "file it names, or run `python "
                    "scripts/calibrate_hardware.py --publish` on this box")
    return {}, (f"no calibration for {gpu_name!r}: none of "
                f"{', '.join(looked)} exists under {directory or HARDWARE_DIR}. "
                "Run `python scripts/calibrate_hardware.py --publish` on this "
                "box, which writes it")


def reference_clock_from_doc(doc: dict, card: str, family: str) -> ReferenceClock:
    """The reference clock of one family, read out of a parsed calibration.

    The walk `reference_clock` performs, split out so a calibration already in
    hand (a yaml beside a published arm, in `recompute`) is read by the SAME
    rule as the one the driver resolves on the pod, rather than by a second
    copy of the field order. `doc` must be a mapping; the caller decided which
    file it is and whether the roof came from it.
    """
    if family not in CLOCK_FIELDS_BY_FAMILY:
        raise ValueError(
            f"{family!r} is not a reference family; one of {REFERENCE_FAMILIES}")
    profile = str(doc.get("name") or "")
    for path, what, grade in CLOCK_FIELDS_BY_FAMILY[family]:
        value = _dig(doc, path)
        if value:
            # A settle polled through the forked fallback is the idle artefact
            # in a longer coat; say so where the source is read, not later.
            if grade == REFERENCE_SETTLE_PLATEAU:
                polled = _dig(doc, ("detail", "settle", "clock_source"))
                if polled and polled != "nvml":
                    what += (f"; and the settle was polled through {polled!r}, "
                             "a forked reader whose samples land tens of "
                             "milliseconds after each synchronise")
            if grade == REFERENCE_UNDER_LOAD:
                polled = _dig(doc, path[:-1] + ("source",))
                if polled and polled != "nvml":
                    what += f"; polled through {polled!r}"
            return ReferenceClock(
                float(value),
                f"{profile}: {'.'.join(path)} = {float(value):.0f} MHz, {what} "
                f"({family} GEMM)",
                card=card, profile=profile, grade=grade, family=family)
    return ReferenceClock(
        None,
        f"{profile} carries no {family} GEMM clock at all: none of "
        f"{', '.join('.'.join(p) for p, _, _ in CLOCK_FIELDS_BY_FAMILY[family])} "
        f"is set, so LEVEL cannot be scored against it for a {family}-family "
        "row. Recalibrate with `python scripts/calibrate_hardware.py --publish`",
        card=card, profile=profile, family=family)


def reference_clock(gpu_name: str | None = None,
                    directory: Path | None = None,
                    family: str = BF16_FAMILY) -> ReferenceClock:
    """The clock THIS card's roof was measured at, from THIS card's calibration.

    PER DTYPE FAMILY since 2026-09-08: `family` names the GEMM whose clock is
    the reference, one of `REFERENCE_FAMILIES`, the bf16 one unless the caller
    asks for the fp8 GEMM's, because the roof an fp8 row is scored against was
    measured at that GEMM's own clock and the committed H200 file has the two
    390 MHz apart. A FAMILY and not a dtype, so a caller has to go through
    `reference_family(dtype)` and cannot hand a format name to a function
    that would quietly read it as "not fp8". Every caller before that date
    meant the bf16 GEMM and still gets it.

    The LEVEL verdict is why `TIMING_BASIS` left v1. The retired throttle check
    compared two IDLE-INSTANT samples either side of a cell, so it detected
    whether the first sample caught the idle boost rather than whether the card
    throttled under load, and on the published alpha-0558 arm it flagged 91% of
    vLLM rows above T=4096 while flagged and unflagged replicates of the same
    cell timed at ratio 0.998. LEVEL replaced it by asking the only question
    that means anything against a roof: was the card at the clock the ROOF was
    measured at while this cell ran. Without a reference that question has no
    left-hand side and `clock_level_ok` is None on every row.

    Returns `mhz=None` WITH A REASON rather than raising, because there are two
    genuinely different worlds here and the caller has to tell them apart:
    `card` empty is a laptop, where no clock is being sampled anyway, and `card`
    set with `mhz` None is a pod holding a card whose ruler was never measured,
    which is a refusal the caller should make before spending a minute on it.
    """
    if gpu_name is None:
        gpu_name = current_gpu_name()
    if family not in REFERENCE_FAMILIES:
        raise ValueError(
            f"{family!r} is not a reference family; one of {REFERENCE_FAMILIES}. "
            "Map a dtype with reference_family() first")
    doc, reason = measured_doc(gpu_name, directory)
    if not doc:
        return ReferenceClock(None, reason, card=gpu_name or "", family=family)
    return reference_clock_from_doc(doc, gpu_name or "", family)


def hardware_for_rows(name: str, rows, allow_unverified: bool = False,
                      directory: Path | None = None) -> Hardware:
    """Resolve a hardware NAME against the rows it will be scored against.

    "measured" stopped being a filename when calibrations went per device, and
    every caller passing that literal broke. scripts/plot.py died on it after a
    three-hour sweep, with all the rows already on disk, which is the worst
    moment to discover a rename.

    Resolution uses the device the ROWS record, not the device running this
    process: figures get drawn on a laptop from a committed CSV, and a published
    result set must keep being plottable against its own roof.
    """
    if name != "measured":
        return load_hardware(name, allow_unverified=allow_unverified,
                             directory=directory)
    gpu = next((r.get("gpu_name", "") for r in rows if r.get("gpu_name")), "")
    hw = load_measured(gpu, directory=directory)
    if hw is None:
        raise FileNotFoundError(
            f"no calibration for {gpu or 'the device these rows name'}; expected "
            f"{measured_slug(gpu) if gpu else 'measured_<device>'}.yaml. Run "
            "scripts/calibrate_hardware.py on that machine, or pass an explicit "
            "--hardware profile.")
    return hw


def efficiency(hw: Hardware, dtype: str, arithmetic_intensity: float,
               achieved_flops_s: float) -> float:
    """Achieved FLOP/s as a fraction of what the roofline permits at this AI.

    The denominator is the roof AT THIS INTENSITY, not peak compute. A
    memory-bound kernel at 95% of its roofline is also at ~4% of peak compute,
    and only the first of those two numbers says anything about the kernel.

    Direction of the modelling error, stated so a reader does not have to guess:
    the intensity comes from COMPULSORY traffic, so it is an upper bound on true
    intensity, so `attainable` is an upper bound on the true roof, so this
    efficiency is UNDERSTATED. That is the conservative direction.
    """
    roof = hw.attainable(dtype, arithmetic_intensity)
    return achieved_flops_s / roof if roof > 0 else 0.0


def plot(rows, out_path, hardware: str = "measured", dtype: str = "bf16",
         label_col: str = "impl", allow_unverified: bool = False):
    """Scatter measured points against the roof. `rows` are schema.Row dicts.

    `hardware` defaults to the measured calibration, NOT to a datasheet part.
    It previously defaulted to "h200_nvl", so any direct library call on an
    H200 SXM roofed its rows against NVL's 835.5 TFLOP/s instead of 989.5 and
    overstated compute efficiency by 18%. scripts/plot.py routes through
    for_device and was never affected; a notebook calling plot() was.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    hw = hardware_for_rows(hardware, rows, allow_unverified=allow_unverified)

    # Only correctness-passing rows are plotted. A row that failed the oracle
    # leaves the driver with its timing zeroed, so this is redundancy rather
    # than the only defence.
    rows = [r for r in rows
            if r.get("dtype") == dtype
            and SC.passed(r)
            and SC.row_float(r, "arith_intensity_compulsory") > 0
            and SC.row_float(r, "tflops") > 0]
    if not rows:
        raise ValueError(
            f"no correctness-passing rows with dtype={dtype} and a positive intensity")

    ai = np.array([SC.row_float(r, "arith_intensity_compulsory") for r in rows])
    tf = np.array([SC.row_float(r, "tflops") for r in rows])

    x = np.logspace(np.log10(max(ai.min() / 4, 1e-2)),
                    np.log10(max(ai.max() * 4, hw.ridge_point(dtype) * 4)), 400)
    roof = np.minimum(hw.peak(dtype), x * hw.bandwidth_bytes_s) / 1e12

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.loglog(x, roof, "k-", lw=1.6, label=f"{hw.name} {dtype} roof")
    ax.axvline(hw.ridge_point(dtype), color="0.6", ls="--", lw=1,
               label=f"ridge {hw.ridge_point(dtype):.0f} FLOP/byte")

    # Series must not fold the timing modes together: the L2-flush axis moves
    # small-batch results by more than most kernel optimisations do, so points
    # measured with and without a flush sit at the SAME x with different y.
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    labels = sorted({SC.series_label(r, label_col) for r in rows})
    for i, label in enumerate(labels):
        m = [j for j, r in enumerate(rows) if SC.series_label(r, label_col) == label]
        ax.loglog(ai[m], tf[m], markers[i % len(markers)], ms=5, alpha=0.85,
                  label=label)

    ax.set_xlabel("compulsory arithmetic intensity (FLOP / byte, UPPER bound)")
    ax.set_ylabel("achieved TFLOP/s")
    ax.set_title(f"MoE grouped GEMM roofline, {hw.name}, {dtype}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
