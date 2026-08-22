"""Run benchmarks inside another virtualenv and merge the results.

vLLM and SGLang each pin their own torch. Installing both alongside your kernel
environment is a coin flip, and it is not a flip worth taking on a box billed by
the minute. Each framework therefore gets its own venv, this module shells into
them, and every process writes its own CSV which schema.merge_csvs combines.

The `env` field on every registered span already says which venv owns it, and
pipeline validation already refuses a tiling that mixes two frameworks, so
nothing here needs new concepts.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_VENV_ROOT = Path(os.environ.get("MOE_VENV_ROOT", "/workspace/venvs"))


class EnvMissing(RuntimeError):
    pass


def venv_python(env: str, root: Path | None = None) -> Path:
    """Interpreter for a named environment. `base` falls back to this process."""
    root = root or DEFAULT_VENV_ROOT
    if env == "base" and not (root / "base").exists():
        return Path(sys.executable)
    p = root / env / "bin" / "python"
    if not p.exists():
        raise EnvMissing(
            f"no interpreter at {p}. Run scripts/setup_runpod.sh to build the "
            f"{env!r} environment, or set MOE_VENV_ROOT.")
    return p


@dataclass
class EnvResult:
    env: str
    returncode: int
    csv: Path | None
    manifest: Path | None
    run_id: str | None
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.csv is not None


def run_env(env: str, cli_args: list[str], root: Path | None = None,
            cwd: Path | None = None, timeout: float | None = None,
            echo: bool = True) -> EnvResult:
    """Invoke `python -m moe.bench.cli` inside `env` and capture what it wrote."""
    py = venv_python(env, root)
    cmd = [str(py), "-m", "moe.bench.cli", "--env", env, *cli_args]
    if echo:
        print(f"[runner] {env}: {shlex.join(cmd)}", flush=True)

    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True,
                          capture_output=True, timeout=timeout)
    if echo and proc.stdout:
        print(proc.stdout, end="", flush=True)
    if proc.returncode != 0 and echo:
        print(proc.stderr, file=sys.stderr, flush=True)

    payload = _last_json_object(proc.stdout)
    return EnvResult(
        env=env,
        returncode=proc.returncode,
        csv=Path(payload["csv"]) if payload and "csv" in payload else None,
        manifest=Path(payload["manifest"]) if payload and "manifest" in payload else None,
        run_id=payload.get("run_id") if payload else None,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def _last_json_object(text: str) -> dict | None:
    """The CLI prints a JSON summary as its final line. Framework imports are
    noisy, so scan from the end rather than trusting the first line."""
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def run_envs(envs: list[str], cli_args: list[str], merged_out: Path,
             root: Path | None = None, cwd: Path | None = None,
             timeout: float | None = None) -> tuple[Path, list[EnvResult]]:
    """Run the same benchmark arguments in several environments and merge.

    An environment that is not installed, or that fails, is reported and
    skipped. One broken framework must not cost you the rest of the session.
    """
    from ..bench.schema import merge_csvs

    results: list[EnvResult] = []
    for env in envs:
        try:
            results.append(run_env(env, cli_args, root=root, cwd=cwd, timeout=timeout))
        except EnvMissing as e:
            print(f"[runner] skipping {env}: {e}", flush=True)
            results.append(EnvResult(env, 127, None, None, None, "", str(e)))
        except subprocess.TimeoutExpired as e:
            print(f"[runner] {env} exceeded its timeout", flush=True)
            results.append(EnvResult(env, 124, None, None, None, "", str(e)))

    produced = [r.csv for r in results if r.ok and r.csv and r.csv.exists()]
    if not produced:
        raise RuntimeError(
            "no environment produced results; see the output above. "
            f"attempted: {[r.env for r in results]}")

    n = merge_csvs(produced, merged_out)
    print(f"[runner] merged {n} rows from {len(produced)} environment(s) "
          f"-> {merged_out}", flush=True)
    return merged_out, results
