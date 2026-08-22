"""Roofline analysis over a results CSV.

Reads peaks from a cited hardware file and refuses to draw an uncited roof.
Arithmetic intensity comes from bytes_model, which computes it per tiling, so a
fused pipeline sits at a different x position than an unfused one and the
predicted benefit of a fusion is readable off the plot before it is measured.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


def efficiency(hw: Hardware, dtype: str, arithmetic_intensity: float,
               achieved_flops_s: float) -> float:
    """Achieved FLOP/s as a fraction of what the roofline permits at this AI.

    This is the honest efficiency number: a memory-bound kernel hitting 95% of
    its roofline is excellent even at 4% of peak compute, and reporting it as
    "4% of peak" would be misleading.
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

    def _num(r, key):
        try:
            return float(r.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    # Only correctness-passing rows are plotted. A row that failed the oracle
    # carries no timing and must never appear as a performance point.
    rows = [r for r in rows
            if r.get("dtype") == dtype
            and str(r.get("correctness_passed", "True")) in ("True", "true", "1")
            and _num(r, "arithmetic_intensity") > 0
            and _num(r, "tflops") > 0]
    if not rows:
        raise ValueError(
            f"no correctness-passing rows with dtype={dtype} and a positive intensity")

    ai = np.array([_num(r, "arithmetic_intensity") for r in rows])
    tf = np.array([_num(r, "tflops") for r in rows])

    x = np.logspace(np.log10(max(ai.min() / 4, 1e-2)),
                    np.log10(max(ai.max() * 4, hw.ridge_point(dtype) * 4)), 400)
    roof = np.minimum(hw.peak(dtype), x * hw.bandwidth_bytes_s) / 1e12

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.loglog(x, roof, "k-", lw=1.6, label=f"{hw.name} {dtype} roof")
    ax.axvline(hw.ridge_point(dtype), color="0.6", ls="--", lw=1,
               label=f"ridge {hw.ridge_point(dtype):.0f} FLOP/byte")

    for label in sorted({r.get(label_col, "") for r in rows}):
        m = [i for i, r in enumerate(rows) if r.get(label_col, "") == label]
        ax.loglog(ai[m], tf[m], "o", ms=5, alpha=0.85, label=label)

    ax.set_xlabel("arithmetic intensity (FLOP / byte, per tiling)")
    ax.set_ylabel("achieved TFLOP/s")
    ax.set_title(f"MoE grouped GEMM roofline, {hw.name}, {dtype}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
