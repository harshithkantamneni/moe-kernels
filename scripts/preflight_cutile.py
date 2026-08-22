#!/usr/bin/env python
"""Decide in minutes whether cuTile is worth any more of this pod's time.

cuTile is real, generally available, and supports Hopper as of cuTile 1.4.0 /
CUDA Tile IR 13.3 (2026-05-26). What it does NOT have is any published
performance data on Hopper for any workload: NVIDIA's TileGym kernel suite
validates only Blackwell and Ampere, its autotune tables carry no sm_90 block,
and every published cuTile number is tagged with a version that predates Hopper
support. So you are not benchmarking a known quantity, you are producing the
first datapoint, and the first thing to establish is whether its Hopper codegen
reaches the tensor-core fast path at all.

    python scripts/preflight_cutile.py                    # environment only
    python scripts/preflight_cutile.py --kernel mod:fn    # + the SASS gate

NOTHING IN THIS SCRIPT HAS BEEN RUN AGAINST A GPU. It is written from verified
documentation and source, not from execution. Treat a failure as "check the
assumption" rather than "the box is broken".
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess

MIN_DRIVER = 580


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return p.returncode, (p.stdout + p.stderr).strip()
    except (OSError, subprocess.SubprocessError) as e:
        return 127, str(e)


def check_driver() -> bool:
    """cuTile hard-requires r580+. A CUDA 12-era driver means it cannot run at
    all, independently of Hopper support, so this is the cheapest possible gate."""
    rc, out = _run(["nvidia-smi", "--query-gpu=driver_version,name",
                    "--format=csv,noheader"])
    if rc != 0:
        print(f"  driver        FAIL  nvidia-smi unavailable: {out[:80]}")
        return False
    first = out.splitlines()[0]
    version, _, name = first.partition(",")
    try:
        major = int(version.strip().split(".")[0])
    except ValueError:
        print(f"  driver        UNKNOWN  could not parse {version!r}")
        return False
    ok = major >= MIN_DRIVER
    print(f"  driver        {'OK  ' if ok else 'FAIL'}  {version.strip()} "
          f"(need r{MIN_DRIVER}+) on{name}")
    if not ok:
        print("                -> destroy this pod and get one with a newer driver")
    return ok


def check_arch() -> bool:
    try:
        import torch
    except ImportError:
        print("  arch          SKIP  no torch in this environment")
        return True
    if not torch.cuda.is_available():
        print("  arch          FAIL  no CUDA device")
        return False
    cap = torch.cuda.get_device_capability(0)
    ok = cap >= (9, 0)
    print(f"  arch          {'OK  ' if ok else 'FAIL'}  sm_{cap[0]}{cap[1]} "
          f"({torch.cuda.get_device_name(0)})")
    if cap == (9, 0):
        print("                -> Hopper: needs cuTile >= 1.4.0 and Tile IR >= 13.3")
    return ok


def check_package() -> bool:
    """The import namespace is `cuda.tile`; the distribution is `cuda-tile`."""
    try:
        mod = importlib.import_module("cuda.tile")
    except ImportError as e:
        print(f"  cuda.tile     FAIL  {e}")
        print("                -> pip install -r requirements/cutile.txt")
        print("                -> do NOT pip install cutile / cutile-python "
              "(PyPI placeholders)")
        return False
    version = getattr(mod, "__version__", "unknown")
    print(f"  cuda.tile     OK    version {version}")
    if version != "unknown" and version < "1.4":
        print("                -> WARNING: Hopper support landed in 1.4.0")
    return True


def check_compiler() -> bool:
    """tileiras is a RUNTIME dependency: cuTile JIT-compiles by invoking it."""
    found = shutil.which("tileiras")
    for dist in ("nvidia-cuda-tileiras", "nvidia-cuda-nvcc", "nvidia-nvvm"):
        try:
            from importlib.metadata import version
            print(f"  {dist:<24} {version(dist)}")
        except Exception:  # noqa: BLE001
            print(f"  {dist:<24} not installed")
    if found:
        rc, out = _run([found, "--version"])
        print(f"  tileiras      OK    on PATH at {found}: {out.splitlines()[0][:60]}"
              if rc == 0 else f"  tileiras      found at {found} but --version failed")
    else:
        print("  tileiras      not on PATH (fine if the pip packages above are "
              "present and agree on major.minor)")
    print("                -> discovery order: pip packages, then PATH, then "
          "$CUDA_HOME/bin, then default CTK paths")
    return True


def sass_gate(target: str, sm: str = "sm_90") -> bool:
    """The decisive question: does cuTile's Hopper codegen reach WGMMA?

    Export your kernel to a cubin and read the SASS. GMMA/HGMMA means it is on
    Hopper's tensor-core fast path. Only HMMA.16816 means it is emitting the
    older mma path, and any benchmark you run measures codegen maturity rather
    than your kernel design. That distinction is worth one pod-hour to settle,
    and "cuTile Hopper codegen is not yet on the WGMMA path" is a publishable
    finding in its own right.
    """
    print(f"\nSASS gate ({target} at {sm})")
    if not shutil.which("cuobjdump"):
        print("  cuobjdump not found; it ships with the CUDA toolkit")
        return False
    print("  This needs YOUR kernel, because a vector-add emits no MMA at all.")
    print("  Export a cubin, then:")
    print("    cuobjdump -sass k.cubin | grep -oE '(HGMMA|GMMA|QGMMA|HMMA|IMMA)[^ ]*' "
          "| sort | uniq -c")
    print("    cuobjdump -sass k.cubin | grep -c 'UTMALDG\\|TMA'")
    print("  GMMA/HGMMA present  -> on the fast path, proceed")
    print("  only HMMA.16816     -> STOP, you are measuring codegen not design")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", default=None,
                    help="module:function of YOUR kernel, for the SASS gate")
    ap.add_argument("--sm", default="sm_90")
    args = ap.parse_args()

    print("cuTile preflight\n")
    results = [check_driver(), check_arch(), check_package(), check_compiler()]
    if args.kernel:
        results.append(sass_gate(args.kernel, args.sm))

    ok = all(results)
    print(f"\n{'PASS' if ok else 'STOP'}: "
          + ("environment looks usable; run the SASS gate next"
             if ok else "fix the FAIL lines above before spending more time"))
    print("Three mature Hopper grouped-GEMM baselines need none of this: "
          "torch grouped_mm (already installed), CuTe DSL, DeepGEMM.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
