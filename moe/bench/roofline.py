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


class UnverifiedHardware(RuntimeError):
    pass


@dataclass(frozen=True)
class Hardware:
    name: str
    bandwidth_bytes_s: float
    peak_flops: dict[str, float]
    source: str

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
    norm = lambda s: "".join(c for c in s.lower() if c.isalnum())
    a, b = norm(hw.name), norm(gpu_name)
    return a in b or b in a


def available_profiles() -> list[str]:
    return sorted(p.stem for p in HARDWARE_DIR.glob("*.yaml"))


def for_device(gpu_name: str) -> str | None:
    """Pick the hardware profile describing this GPU, or None if none does.

    Keeps the repo from being hardcoded to one part. Silence is the right answer
    when nothing matches: a missing roof is recoverable, a wrong one is not.
    """
    for stem in available_profiles():
        if stem == "measured":
            continue
        try:
            hw = load_hardware(stem, allow_unverified=True)
        except (ValueError, KeyError):
            continue
        if device_matches(hw, gpu_name):
            return stem
    return None


def load_measured() -> Hardware | None:
    """Ceilings measured by scripts/calibrate_hardware.py, if it has been run.

    That script writes measured.yaml in exactly load_hardware's schema, so this
    is a one-line call rather than a second yaml parser. Returns None when the
    calibration has not been run, so callers leave the efficiency columns empty
    instead of quoting a datasheet peak.
    """
    try:
        return load_hardware("measured")
    except (FileNotFoundError, ValueError, KeyError, UnverifiedHardware):
        return None


def efficiency(hw: Hardware, dtype: str, arithmetic_intensity: float,
               achieved_flops_s: float) -> float:
    """Achieved FLOP/s as a fraction of what the roofline permits at this AI.

    This is the honest efficiency number: a memory-bound kernel hitting 95% of
    its roofline is excellent even at 4% of peak compute, and reporting it as
    "4% of peak" would be misleading.

    Direction of the modelling error, stated so a reader does not have to guess:
    the intensity comes from COMPULSORY traffic, so it is an upper bound on true
    intensity, so `attainable` is an upper bound on the true roof, so this
    efficiency is UNDERSTATED. That is the conservative direction.
    """
    roof = hw.attainable(dtype, arithmetic_intensity)
    return achieved_flops_s / roof if roof > 0 else 0.0


def plot(rows, out_path, hardware: str = "h200_nvl", dtype: str = "bf16",
         label_col: str = "impl", allow_unverified: bool = False):
    """Scatter measured points against the roof. `rows` are schema.Row dicts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    hw = load_hardware(hardware, allow_unverified=allow_unverified)

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
