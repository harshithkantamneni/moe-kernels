"""Command line entry point for benchmarking.

Two audiences. On the GPU box, `run_all.sh` calls this to execute a profile.
On a laptop, `--dry-run` builds and validates the entire matrix and reports what
a session would cost, without CUDA and without spending anything.

    python -m moe.bench.cli --profile standard --dry-run
    python -m moe.bench.cli --profile smoke --out-dir /workspace/results

`--models`, `--tokens` and `--routings` override the profile's axes in place, so
a rented box never has to edit `profiles.py` to sweep something the file does
not name. That is not a convenience: an edited working tree stamps `git_dirty`
on every row of the run.

    python -m moe.bench.cli --profile trace-replay \\
        --models deepseek-v2-lite \\
        --routings uniform,trace:deepseek-v2-lite-chat-decode@b3l17

MOE_FORCE_TILE pins the Triton tile for the whole run instead of letting vLLM's
ladder step it with the token count. It is an environment variable rather than a
flag because `scripts/pod_session.sh` and `docs/POD_RUNBOOK.md` already document
it as one, and because it must reach every venv `moe.runner.subproc` spawns:

    MOE_FORCE_TILE='{"BLOCK_SIZE_M":128,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":64,
                     "GROUP_SIZE_M":1,"num_warps":8,"num_stages":4}' \\
        python -m moe.bench.cli --profile crossing-uniform --env vllm \\
               --groups baselines --impl vllm_fused_experts

Exit codes, so a shell can tell a falsified prediction from a broken instrument:
0 ran, 1 the plan is invalid, 2 nothing to benchmark, 3 a tile was forced and no
cell in the run stands under it, 4 a cell ran pinned and its row did not show
the pin. See moe/bench/force_tile.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import moe

from ..spec import MODEL_CONFIGS, RoutingSpec
from ..stages import registry
from . import force_tile as FT
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
    p.add_argument("--routings", default=None,
                   help="comma-separated routing override, in the label form "
                        "the dry run prints and the CSV records: uniform, "
                        "zipf:1.2, dirichlet:0.3, hot:0.5, "
                        "trace:mixtral-8x7b-chat-decode, or a pinned slice "
                        "trace:mixtral-8x7b-chat-decode@b3l17")
    p.add_argument("--max-minutes", type=float, default=None,
                   help="stop cleanly at this wall-clock budget; resume later")
    p.add_argument("--traces-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--groups", default="reference,kernels",
                   help="implementation groups to import: reference,kernels,baselines")
    p.add_argument("--list-impls", action="store_true")
    p.add_argument("--no-pipeline-scope", action="store_true",
                   help="drop the whole-layer reference cells `full` adds. They "
                        "time a python loop over every expert and dominate the "
                        "run, and they measure the reference rather than any "
                        "kernel, so a kernel comparison wants full's token grid "
                        "and routings without them")
    p.add_argument("--include-reference", action="store_true",
                   help="also benchmark the reference spans; useful to size "
                        "the matrix before any kernel exists, and to get a "
                        "deliberately slow lower bound")
    return p.parse_args(argv)


def parse_routing(text: str) -> RoutingSpec:
    """Read one routing off the command line, in the form everything prints it.

    `RoutingSpec.label` is the format: `uniform`, `zipf:1.2`, `trace:<id>`. So
    what a dry run printed, or what a published CSV recorded, can be pasted back
    in without translation, and there is one spelling of a routing in this
    project rather than two.

    WHY THIS EXISTS. `kind="trace"` is the axis the repo calls its
    differentiator, and until `--routings` there was no way to point a sweep at
    a capture except by editing `profiles.py` on the box -- which stamps
    `git_dirty` on every row of the run and makes the arm unpublishable.

    Errors exit rather than raise: this runs during argument parsing, where a
    traceback is noise and a misspelt routing should cost nothing.
    """
    kind, _, rest = text.strip().partition(":")
    kind, rest = kind.strip(), rest.strip()
    if kind == "trace":
        if not rest:
            raise SystemExit("routing 'trace' needs an id, e.g. "
                             "trace:mixtral-8x7b-chat-decode")
        return RoutingSpec("trace", trace_id=rest)
    param = 0.0
    if rest:
        try:
            param = float(rest)
        except ValueError:
            raise SystemExit(f"routing {text!r}: {rest!r} is not a number") from None
    try:
        return RoutingSpec(kind, param)
    except ValueError as e:
        raise SystemExit(f"routing {text!r}: {e}") from None


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
    if getattr(args, "routings", None):
        routings = tuple(parse_routing(r) for r in args.routings.split(",")
                         if r.strip())
        if not routings:
            raise SystemExit("--routings was given but names no routing")
        changes["routings"] = routings
    if getattr(args, "no_pipeline_scope", False):
        changes["include_pipeline_scope"] = False
    profile = replace(profile, **changes) if changes else profile
    # A trace belongs to the model it was captured from, and `Profile.specs`
    # drops the pairings that are not that. An override can therefore ask for a
    # matrix whose every cell is dropped, and finding that out from a run that
    # writes no rows is the expensive way.
    if not profile.specs():
        raise SystemExit(
            f"the overridden matrix has no cells: models {list(profile.models)} "
            f"against routings {[r.label for r in profile.routings]}. A trace "
            "is only valid on the model it was captured from.")
    return profile


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


def force_tile_plan(forced: FT.ForcedTile | None, impls) -> tuple[list[str], bool]:
    """What the pin would do to this plan, printed BEFORE anything is spent.

    Returns the lines and whether the plan is honourable at all. A plan where
    nothing can be pinned is not a warning: it is the exact state the
    2026-09-01 session ran in for 54 minutes, and the dry run is the review step
    that was supposed to catch it.

    Off the GPU box no framework span registers, so a laptop dry run always
    reports "nothing can pin this". That is honest rather than a nuisance: it is
    also what the run would do on a pod where the vLLM import failed.
    """
    if forced is None:
        return [], True
    can, cannot = FT.split_by_pinnability(impls)
    lines = [f"\nforced tile     {forced.describe()}",
             f"  resume key    {forced.fingerprint()} (in every manifest key, "
             f"so a pinned run cannot resume an unpinned one)"]
    if can:
        lines.append(f"  CAN PIN       {', '.join(can)}")
    if cannot:
        lines.append(f"  CANNOT PIN    {', '.join(cannot)}")
        lines.append("                those cells are RECORDED AND SKIPPED, "
                     "never measured unpinned")
    if not can:
        lines.append(
            f"\nREFUSED: {FT.ENV_VAR} is set and no implementation in this plan "
            f"can honour it, so the run would measure nothing pinned. Either "
            f"unset it, or point the sweep at the vLLM span: --env vllm "
            f"--groups baselines --impl vllm_fused_experts, inside the vllm "
            f"venv. Off the GPU box no framework span registers at all.")
    return lines, bool(can)


def dry_run(profile: PR.Profile, args, traces, forced=None) -> int:
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
    # The axes above are a cartesian product and `Profile.specs` is not: a trace
    # only sweeps the model it was captured from. Say so, or the two numbers
    # disagree on the page with nothing to explain them.
    product = (len(profile.models) * len(profile.token_counts)
               * len(profile.dtypes) * len(profile.routings)
               * len(profile.seeds))
    dropped = product - p.specs
    print(f"specs           {p.specs}" + (
        f"   ({dropped} of {product} dropped: a trace is only valid on the "
        "model it was captured from)" if dropped else ""))
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

    # `registry()[name]` rather than a second call to candidate_impls, so this
    # cannot describe a different implementation set from the one just planned.
    lines, honourable = force_tile_plan(forced, [registry()[n] for n in p.impls])
    for line in lines:
        print(line)

    print("\nNo GPU was used. Nothing was spent.")
    return 0 if (p.ok and honourable) else 1


def main(argv=None) -> int:
    args = parse_args(argv)

    # BEFORE bootstrap, which imports vLLM and costs ~20 s on the pod: a
    # mistyped tile config should cost nothing at all. SystemExit rather than a
    # traceback for the reason parse_routing gives -- this is argument handling,
    # and the message is the useful part.
    try:
        forced = FT.from_env()
    except FT.ForceTileMalformed as e:
        # str(e), not a wrapped message: every refusal in force_tile already
        # names the variable and says what to do instead.
        raise SystemExit(str(e)) from None

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
        return dry_run(profile, args, traces, forced)

    cfg_kw = dict(out_dir=args.out_dir, env_name=args.env, device=args.device,
                  warmup=profile.warmup, trials=profile.trials,
                  iters=profile.iters, l2_modes=profile.l2_modes,
                  graph_modes=profile.graph_modes, routing_info=routing_info,
                  force_tile=forced, **measured_ceilings())
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

    if forced is not None:
        # BEFORE the sweep, and a refusal here costs nothing: the alternative is
        # discovering after 22 GB of weights and an fp32 oracle that the plan
        # contained no span able to honour the pin, which is the shape of the
        # 2026-09-01 session.
        lines, honourable = force_tile_plan(forced, planned_impls(args))
        for line in lines:
            print(line)
        if not honourable:
            return 3
        print("[cli] gates registered before the run, with their thresholds:")
        for gid, claim, _, threshold, _ in cfg.force_tile_ledger.gates():
            print(f"[cli]   {gid}  {claim}: {threshold}")

    started = time.time()
    path = run_sweep(cells, cfg, routing, info=info)
    print(f"[cli] run_id={cfg.run_id} elapsed={time.time() - started:.1f}s")
    # The JSON line first: moe.runner.subproc parses it out of stdout and a
    # non-zero exit below must not cost the caller the paths it needs to resume.
    print(json.dumps({"csv": str(path), "manifest": str(cfg.manifest_path),
                      "run_id": cfg.run_id}))
    return force_tile_verdict(forced, cfg.force_tile_ledger)


def planned_impls(args) -> list:
    """The spans this invocation would benchmark, filtered exactly as
    `profiles.cells` filters them, so the pinning report cannot describe an
    implementation set the sweep does not use."""
    impls = PR.candidate_impls(env=args.env,
                               include_reference=args.include_reference)
    keep = tuple(args.impl)
    return [s for s in impls if not keep or s.name in keep]


def force_tile_verdict(forced, ledger) -> int:
    """The run's exit code, once the pin has been accounted for.

    NON-VACUITY, and it is the whole reason the ledger counts anything. A sweep
    that was asked to pin and pinned nothing writes rows nobody can quote as
    pinned, and every check downstream of it examines zero pinned rows and
    reports zero failures -- which is exactly how MOE_FORCE_TILE stayed set for
    a whole session with nothing reading it. So that state exits non-zero and
    says which of the two shapes it is.
    """
    if forced is None:
        return 0
    for gid, claim, measured, threshold, ok in ledger.gates():
        print(f"[force-tile] GATE {gid}  {claim}: {measured} against "
              f"{threshold}  {'PASS' if ok else 'FAIL'}")
    if ledger.unobserved:
        print(f"REFUSED (4): {len(ledger.unobserved)} implementation(s) ran "
              f"under {FT.ENV_VAR} and produced no row showing that tile. No "
              f"number from this run may be quoted as tile-pinned. See the NOT "
              f"HONOURED lines above.")
        return 4
    if ledger.vacuous():
        print(f"REFUSED (3): {FT.ENV_VAR} was set and not one cell of this run "
              f"stands under it ({ledger.skipped_cells} cells were skipped as "
              f"unpinnable). Point the sweep at the vLLM span -- --env vllm "
              f"--groups baselines --impl vllm_fused_experts, inside the vllm "
              f"venv -- or unset the variable to sweep vLLM's own ladder.")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
