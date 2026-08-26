"""Roofline analysis over a results CSV.

Reads peaks from a cited hardware file and refuses to draw an uncited roof.
Arithmetic intensity comes from bytes_model, which computes it per tiling, so a
fused pipeline sits at a different x position than an unfused one and the
predicted benefit of a fusion is readable off the plot before it is measured.
"""
from __future__ import annotations

from dataclasses import dataclass
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
    return Hardware(
        name=data["name"],
        bandwidth_bytes_s=float(bw) * 1e12,
        peak_flops={k: v for k, v in peaks.items() if v},
        source=data.get("source", ""),
        ceiling_pattern=(data.get("detail") or {}).get("ceiling_pattern", ""),
        tdp_w=data.get("tdp_w"),
    )


def peak_bandwidth(name: str = "h200_nvl") -> float | None:
    """Bandwidth for cost prediction. Returns None if unavailable rather than
    guessing, and callers fall back to doing the measurement."""
    try:
        return load_hardware(name, allow_unverified=True).bandwidth_bytes_s
    except (FileNotFoundError, ValueError, KeyError):
        return None


def device_matches(hw: Hardware, gpu_name: str) -> bool:
    """Does this hardware profile describe the GPU the rows were measured on?

    Plotting an H100 run against an H200 roof would understate efficiency by
    the ratio of their peaks and is exactly the kind of error this repo exists
    not to make. Comparison is loose on purpose: datasheet names ("NVIDIA H200
    NVL") and torch's device names ("NVIDIA H200 NVL") agree on the part but
    not always on spacing or suffixes.
    """
    if not gpu_name:
        return True                      # nothing to check against
    def norm(text: str) -> str:
        return "".join(c for c in text.lower() if c.isalnum())

    a, b = norm(hw.name), norm(gpu_name)
    return a in b or b in a


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


def measured_slug(gpu_name: str) -> str:
    """Filename stem for this device's calibration.

    One harness, one calibration per device. A single shared `measured.yaml`
    meant calibrating on a second GPU overwrote the first, and a later re-plot
    of the published sweep then scored it against the wrong roof.
    """
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
