"""Command line entry point for benchmarking.

Two audiences. On the GPU box, `run_all.sh` calls this to execute a profile.
On a laptop, `--dry-run` builds and validates the entire matrix and reports what
a session would cost, without CUDA and without spending anything.

    python -m moe.bench.cli --profile standard --dry-run
    python -m moe.bench.cli --profile smoke --out-dir /workspace/results
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import moe

from ..spec import MODEL_CONFIGS
from . import profiles as PR
from . import timing as T
from .driver import RunConfig, run_sweep


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="moe.bench.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", default="smoke", choices=sorted(PR.PROFILES))
    p.add_argument("--dry-run", action="store_true",
                   help="validate the matrix and report its size; no GPU needed")
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("--run-id", default=None,
                   help="reuse a previous run id to resume its manifest")
    p.add_argument("--env", default="base",
                   help="which virtualenv this process is; recorded in every row")
    p.add_argument("--impl", action="append", default=[],
                   help="restrict to these implementations (repeatable)")
    p.add_argument("--models", default=None,
                   help="comma-separated model override, e.g. mixtral-8x7b,toy")
    p.add_argument("--tokens", default=None,
                   help="comma-separated token-count override")
    p.add_argument("--max-minutes", type=float, default=None,
                   help="stop cleanly at this wall-clock budget; resume later")
    p.add_argument("--traces-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--groups", default="reference,kernels",
                   help="implementation groups to import: reference,kernels,baselines")
    p.add_argument("--list-impls", action="store_true")
    p.add_argument("--include-reference", action="store_true",
                   help="also benchmark the reference spans; useful to size "
                        "the matrix before any kernel exists, and to get a "
                        "deliberately slow lower bound")
    return p.parse_args(argv)


def apply_overrides(profile: PR.Profile, args) -> PR.Profile:
    from dataclasses import replace
    changes = {}
    if args.models:
        names = tuple(m.strip() for m in args.models.split(",") if m.strip())
        unknown = [m for m in names if m not in MODEL_CONFIGS]
        if unknown:
            raise SystemExit(f"unknown model(s) {unknown}; known: {sorted(MODEL_CONFIGS)}")
        changes["models"] = names
    if args.tokens:
        changes["token_counts"] = tuple(int(t) for t in args.tokens.split(","))
    return replace(profile, **changes) if changes else profile


def measured_ceilings() -> dict:
    """RunConfig kwargs carrying this machine's measured ceilings, if any."""
    from .roofline import load_measured
    hw = load_measured()
    return {"hardware": hw} if hw is not None else {}


def env_version(env: str) -> str:
    """Version of the framework this environment provides.

    Publishing "we beat vLLM's fused_moe" without the vLLM version in the row is
    not reproducible, so the version is a column and not a footnote.
    """
    if env == "base":
        try:
            import torch
            import triton
            return f"torch {torch.__version__} / triton {triton.__version__}"
        except ImportError:
            return ""
    try:
        from importlib.metadata import PackageNotFoundError, version
        return version(env)
    except (ImportError, PackageNotFoundError):
        return ""


def build_routing_source(args):
    """Resolve routing for each cell, loading traces only if a cell needs them."""
    from ..routing.distributions import routing_source
    from ..routing.traces import TRACE_DIR, TraceSet

    directory = args.traces_dir or TRACE_DIR
    traces = TraceSet.load(directory) if Path(directory).exists() else TraceSet({})

    def source(spec):
        return routing_source(spec, device=args.device, traces=traces)

    def provenance(spec):
        return traces.provenance(spec)

    return source, provenance, traces


def time_limited(cells, max_minutes: float | None):
    """Stop yielding cells once the budget is spent.

    The check happens between cells, so a running cell always finishes and its
    row is written. The manifest makes the next session resume from here.
    """
    if not max_minutes:
        yield from cells
        return
    deadline = time.monotonic() + max_minutes * 60.0
    stopped = False
    for cell in cells:
        if time.monotonic() >= deadline:
            stopped = True
            break
        yield cell
    if stopped:
        print(f"[cli] wall-clock budget of {max_minutes:g} min reached; "
              "stopping cleanly. Re-run with the same --run-id to resume.")


def dry_run(profile: PR.Profile, args, traces) -> int:
    """Print what a sweep would do. All computation lives in profiles.plan()."""
    p = PR.plan(profile, env=args.env, impl_filter=tuple(args.impl),
                include_reference=args.include_reference, traces=traces)

    print(f"profile         {profile.name}")
    print(f"env             {args.env}")
    print(f"models          {', '.join(profile.models)}")
    print(f"token counts    {', '.join(str(t) for t in profile.token_counts)}")
    print(f"dtypes          {', '.join(profile.dtypes)}")
    print(f"routings        {', '.join(r.label for r in profile.routings)}")
    print(f"seeds           {', '.join(str(s) for s in profile.seeds)}")
    print(f"specs           {p.specs}")
    print(f"implementations {len(p.impls)}: "
          f"{', '.join(p.impls) if p.impls else 'NONE REGISTERED'}")
    print(f"timing modes    {p.modes} "
          f"(l2_flush={list(profile.l2_modes)}, cuda_graph={list(profile.graph_modes)})")
    print(f"\nplanned cells   {p.planned}")
    print(f"timing rows     {p.timing_rows} (upper bound; a skipped or "
          "uncapturable graph mode still writes one row)")
    print(f"skipped         {p.unsupported} (implementation does not support the cell)")

    from .footprint import worst_cell
    spec, fp = worst_cell(profile.specs())
    if fp is not None:
        print(f"\npeak device memory for the heaviest cell ({spec.label}):")
        print(f"  expert weights  {fp.weights / 1e9:7.1f} GB   (one MoE layer, "
              f"random; nothing is downloaded)")
        print(f"  activations     {fp.activations / 1e9:7.1f} GB")
        print(f"  fp32 oracle     {(fp.oracle + fp.oracle_expert) / 1e9:7.1f} GB")
        print(f"  PEAK            {fp.peak / 1e9:7.1f} GB")
        for label, cap in (("H200 (141 GB)", 141e9), ("H100 NVL (94 GB)", 94e9),
                           ("A100 (80 GB)", 80e9)):
            print(f"    {label:<18} {'fits' if fp.fits_in(cap) else 'DOES NOT FIT'}")

    if p.missing_traces:
        print(f"MISSING TRACES  {list(p.missing_traces)}")
    if p.problems:
        print(f"\nINVALID TILINGS ({len(p.problems)}):")
        for line in p.problems[:20]:
            print(f"  {line}")
        if len(p.problems) > 20:
            print(f"  ... and {len(p.problems) - 20} more")
    if not p.impls:
        print("\nNothing to benchmark: no non-reference implementations are "
              "registered. Write a kernel in moe/kernels/ (see TEMPLATE.md).")

    print("\nNo GPU was used. Nothing was spent.")
    return 0 if p.ok else 1


def main(argv=None) -> int:
    args = parse_args(argv)
    moe.bootstrap(*[g.strip() for g in args.groups.split(",") if g.strip()])

    if args.list_impls:
        for span in PR.candidate_impls(include_reference=True):
            print(f"{span.name:<32} covers={'+'.join(span.covers):<24} "
                  f"env={span.env:<8} graph_safe={span.cuda_graph_safe} "
                  f"dtypes={','.join(span.dtypes)}")
        return 0

    profile = apply_overrides(PR.get(args.profile), args)
    routing, routing_info, traces = build_routing_source(args)

    if args.dry_run:
        return dry_run(profile, args, traces)

    cfg_kw = dict(out_dir=args.out_dir, env_name=args.env, device=args.device,
                  warmup=profile.warmup, trials=profile.trials,
                  iters=profile.iters, l2_modes=profile.l2_modes,
                  graph_modes=profile.graph_modes, routing_info=routing_info,
                  **measured_ceilings())
    if args.run_id:
        cfg_kw["run_id"] = args.run_id
    cfg = RunConfig(**cfg_kw)

    cells = time_limited(
        PR.cells(profile, env=args.env, impl_filter=tuple(args.impl),
                 include_reference=args.include_reference),
        args.max_minutes)

    info = T.runtime_info()
    # NOT info["env_name"]: _base_row sets that from cfg.env_name, and passing
    # both collides. env_version is not one of its arguments, so it belongs here.
    info["env_version"] = env_version(args.env)
    print(f"[cli] env={args.env} {info['env_version']} "
          f"gpu={info.get('gpu_name', 'none')}")

    started = time.time()
    path = run_sweep(cells, cfg, routing, info=info)
    print(f"[cli] run_id={cfg.run_id} elapsed={time.time() - started:.1f}s")
    print(json.dumps({"csv": str(path), "manifest": str(cfg.manifest_path),
                      "run_id": cfg.run_id}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
