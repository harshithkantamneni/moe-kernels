#!/usr/bin/env python3
"""THE KIT'S ENVIRONMENT PREFLIGHT: smoke.sh's and main.sh's first hard step (`env-preflight`, cap 600 s). Every
environment fault the smoke rental of 2026-11-19 met one at a time, deep into the sequence, is checked here in the
first minutes, each with its reason (SMOKE-FIXES.md items 1-5):

  P1  git      the checkout is clean (git status --porcelain empty: the pinned v7p driver refuses a dirty tree,
               git-at-start.txt) and HEAD is the expected commit (--expect-commit, else $PROD_EXPECT_COMMIT,
               else the commit setup_vm.sh wrote at the top of $MOE_HOME/env.sh)
  P2  idle     no compute process on the GPU
  P3  power    enforced.power.limit reads back --power-w (700; power.limit reads the 900 W default setting)
  P4  locks    each --locks clock (1710, 1410) is set (`sudo -n nvidia-smi -lgc F,F` says "GPU clocks set to"
               with F) and restored (`-rgc` exit 0); the card is left at default clocks
  P5  tmpdir   --tmpdir is creatable and writable (the kit exports it as TMPDIR)
  P6  tools    ninja, nsys and ncu on PATH (the kit's PATH: the venv's bin first)
  P7  nsys     `nsys profile --cuda-graph-trace=node /bin/true` under TMPDIR=--tmpdir exits 0 (the VM's
               /tmp/nvidia is root-owned after setup_vm's ncu)
  P8  hf-rev   the run's model at its pinned revision (prodlib.HF_REVISIONS): config.json and the weight index
               downloaded (network allowed here), then serve_capture.resolve_revision (the fixed resolver) gives
               the pin back
  P9  frontier the Frontier repo is reachable and holds its pinned commit (frontier_setup.sh's URL and COMMIT)
  P10 disk     free space under --disk > --min-free-gb (200)
  P11 setup    setup_vm.sh's own preflight said READY ($SESSION_ROOT/PREFLIGHT.txt: `verdict READY`)
  P12 hf-rate  the model's weight download PRICED by a measured throughput (review 2026-11-19 m6): up to
               --rate-bytes (1 GB) or --rate-seconds (20 s) of the first weight shard at the pinned revision
               (the index's weight_map; --rate-url overrides) streamed and discarded; FAILS under --min-mbps
               (50 MB/s), reporting the rate and the whole download's expected time (the index's total_size)

Writes --out (JSON, one record per check) and prints one PASS/FAIL line per check. Exit 0 all PASS, 3 any FAIL.
Commands are found on PATH (nvidia-smi, sudo, git, ninja, ncu) or given (--nsys): the tests put fakes there.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sh(cmd: list[str], timeout: float = 60, env: dict | None = None) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except FileNotFoundError:
        return 127, f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, f"{' '.join(cmd)}: timed out after {timeout:.0f}s"


def expected_commit(arg: str | None, moe_home: Path) -> tuple[str | None, str]:
    if arg:
        return arg, "--expect-commit"
    if os.environ.get("PROD_EXPECT_COMMIT"):
        return os.environ["PROD_EXPECT_COMMIT"], "$PROD_EXPECT_COMMIT"
    env_sh = moe_home / "env.sh"
    if env_sh.is_file():
        m = re.search(r"Written by scripts/setup_vm\.sh at (\S+)\.", env_sh.read_text())
        if m:
            return m.group(1), str(env_sh)
    return None, "none"


def p1_git(repo: Path, expect: str | None, src: str) -> tuple[bool, str]:
    rc, out = sh(["git", "-C", str(repo), "status", "--porcelain"])
    if rc != 0:
        return False, f"git status failed: {out.strip()[:200]}"
    if out.strip():
        return False, ("the checkout is DIRTY (the pinned v7p driver refuses it): "
                       + "; ".join(out.strip().splitlines()[:5]))
    rc, head = sh(["git", "-C", str(repo), "rev-parse", "HEAD"])
    head = head.strip()
    if not expect:
        return False, f"no expected commit (pass --expect-commit or $PROD_EXPECT_COMMIT); HEAD is {head}"
    rc, want = sh(["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", expect + "^{commit}"])
    want = want.strip()
    if rc != 0:
        return False, f"the expected commit {expect} (from {src}) is not in this checkout; HEAD is {head}"
    if want != head:
        return False, f"HEAD {head} is not the expected commit {want} ({expect}, from {src})"
    return True, f"clean at {head} (expected, from {src})"


def p2_idle() -> tuple[bool, str]:
    rc, out = sh(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"])
    if rc != 0:
        return False, f"nvidia-smi --query-compute-apps failed (exit {rc}): {out.strip()[:200]}"
    apps = [x for x in out.splitlines() if x.strip() and "No running" not in x]
    return (not apps), ("no compute process on the GPU" if not apps else "the GPU is busy: " + "; ".join(apps[:5]))


def p3_power(want: int) -> tuple[bool, str]:
    rc, out = sh(["nvidia-smi", "--query-gpu=enforced.power.limit", "--format=csv,noheader,nounits"])
    line = out.strip().splitlines()[0].strip() if out.strip() else ""
    try:
        got = int(float(line))
    except ValueError:
        return False, f"enforced.power.limit unreadable (exit {rc}): {line[:200]!r}"
    return got == want, f"enforced power limit {got} W" + ("" if got == want else f", want {want} W")


def p4_locks(locks: list[int]) -> tuple[bool, str]:
    sh(["sudo", "-n", "nvidia-smi", "-pm", "1"])
    notes = []
    ok = True
    for f in locks:
        rc, out = sh(["sudo", "-n", "nvidia-smi", "-lgc", f"{f},{f}"])
        took = rc == 0 and "GPU clocks set to" in out and str(f) in out
        rrc, rout = sh(["sudo", "-n", "nvidia-smi", "-rgc"])
        if not took:
            ok = False
            notes.append(f"-lgc {f},{f} did not take (exit {rc}): {out.strip()[:160]}")
        else:
            notes.append(f"{f} MHz set")
        if rrc != 0:
            ok = False
            notes.append(f"-rgc after {f} failed (exit {rrc}): {rout.strip()[:160]}")
    return ok, "; ".join(notes) + ("; restored to default clocks" if ok else "")


def p5_tmpdir(tmp: Path) -> tuple[bool, str]:
    try:
        tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=tmp) as f:
            f.write(b"ok")
        return True, f"{tmp} writable"
    except OSError as exc:
        return False, f"{tmp} not writable: {exc}"


def p6_tools(nsys: str) -> tuple[bool, str]:
    found = {t: shutil.which(t) for t in ("ninja", "nsys", "ncu")}
    if not found["nsys"] and nsys and os.access(nsys, os.X_OK):
        found["nsys"] = nsys
    miss = [t for t, p in found.items() if not p]
    return (not miss), ("missing on PATH: " + ", ".join(miss) if miss
                        else ", ".join(f"{t} {p}" for t, p in found.items()))


def p7_nsys(nsys: str, tmp: Path) -> tuple[bool, str]:
    d = tmp / "preflight-nsys"
    d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TMPDIR=str(tmp))
    rc, out = sh([nsys, "profile", "-t", "cuda", "--cuda-graph-trace=node", "--force-overwrite", "true",
                  "-o", str(d / "true"), "/bin/true"], timeout=120, env=env)
    bad = re.search(r"Failed to create directory[^\n]*", out)
    if rc != 0 or bad:
        return False, f"nsys profile of /bin/true exit {rc}: {(bad.group(0) if bad else out.strip()[-200:])}"
    return True, f"nsys profile --cuda-graph-trace=node ran under TMPDIR={tmp}"


def p8_hf(model_key: str) -> tuple[bool, str]:
    sys.path[:0] = [str(HERE)]
    import prodlib as PL
    import serve_capture as S
    m = PL.MODELS.get(model_key)
    if m is None:
        return False, f"unknown model {model_key}"
    rev = PL.hf_revision(model_key)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        return False, f"huggingface_hub not importable: {exc}"
    try:
        hf_hub_download(m.hf_repo, "config.json", revision=rev)
    except Exception as exc:                                          # noqa: BLE001
        return False, f"{m.hf_repo} config.json at {rev}: {type(exc).__name__}: {str(exc)[:200]}"
    errs = []
    for w in S.WEIGHT_FILES[:2]:      # the index (sharded models), else the single safetensors file
        try:
            hf_hub_download(m.hf_repo, w, revision=rev)
            break
        except Exception as exc:                                      # noqa: BLE001
            errs.append(f"{w}: {type(exc).__name__}")
    else:
        return False, f"{m.hf_repo} at {rev}: no weight index or file downloadable ({'; '.join(errs)})"
    got = S.resolve_revision(m.hf_repo, rev)
    if rev:
        return got == rev, (f"{m.hf_repo} resolves to the pin {rev}" if got == rev
                            else f"{m.hf_repo} pinned {rev}, the fixed resolver gives {got}")
    return got is not None, f"{m.hf_repo} UNPINNED, resolves to {got}"


def weight_source(model_key: str) -> tuple[str | None, float | None, str]:
    """(URL of the model's first weight shard at its pin, the weights' total bytes or None, why-not)."""
    sys.path[:0] = [str(HERE)]
    import prodlib as PL
    m = PL.MODELS.get(model_key)
    if m is None:
        return None, None, f"unknown model {model_key}"
    rev = PL.hf_revision(model_key)
    try:
        from huggingface_hub import hf_hub_download, hf_hub_url
    except ImportError as exc:
        return None, None, f"huggingface_hub not importable: {exc}"
    try:
        idx = json.loads(Path(hf_hub_download(m.hf_repo, "model.safetensors.index.json", revision=rev)).read_text())
        shard = sorted(set(idx["weight_map"].values()))[0]
        total = (idx.get("metadata") or {}).get("total_size")
    except Exception:                                                 # noqa: BLE001
        shard, total = "model.safetensors", None
    return hf_hub_url(m.hf_repo, shard, revision=rev), (float(total) if total else None), ""


def p12_rate(model_key: str, url: str | None, max_bytes: float, max_s: float, min_mbps: float) -> tuple[bool, str]:
    import time
    import urllib.request
    total = None
    if not url:
        url, total, why = weight_source(model_key)
        if not url:
            return False, why
    got, t0 = 0, time.monotonic()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "moe-kernels-preflight"})
        with urllib.request.urlopen(req, timeout=30) as r:            # noqa: S310 (https, or file:// in the tests)
            while got < max_bytes and time.monotonic() - t0 < max_s:
                buf = r.read(1 << 23)
                if not buf:
                    break
                got += len(buf)
    except Exception as exc:                                          # noqa: BLE001
        return False, f"{url}: {type(exc).__name__}: {str(exc)[:200]}"
    dt = max(time.monotonic() - t0, 1e-6)
    mbps = got / dt / 1e6
    if got == 0:
        return False, f"{url}: nothing downloaded in {dt:.1f} s"
    price = f"; the weights ({total / 1e9:.1f} GB) would take ~{total / 1e6 / mbps:.0f} s" if total else ""
    msg = f"measured {mbps:.0f} MB/s ({got / 1e6:.0f} MB in {dt:.1f} s){price}"
    if mbps < min_mbps:
        return False, f"{msg}: under the {min_mbps:g} MB/s floor (the {model_key} download would eat the rental)"
    return True, msg


def frontier_pin(setup_sh: Path) -> tuple[str | None, str | None]:
    t = setup_sh.read_text()
    c = re.search(r'^COMMIT="([0-9a-f]+)"', t, re.M)
    u = re.search(r'^URL="([^"]+)"', t, re.M)
    return (u.group(1) if u else None), (c.group(1) if c else None)


def p9_frontier(url: str, commit: str, src: Path, tmp: Path) -> tuple[bool, str]:
    rc, out = sh(["git", "ls-remote", "--heads", url], timeout=60)
    if rc != 0:
        return False, f"{url} unreachable (git ls-remote exit {rc}): {out.strip()[:200]}"
    if (src / ".git").exists():
        sh(["git", "-C", str(src), "fetch", "--quiet", "origin"], timeout=120)
        rc, _ = sh(["git", "-C", str(src), "cat-file", "-e", commit + "^{commit}"])
        return rc == 0, (f"{src} holds {commit}" if rc == 0 else f"{src} does not hold {commit} after a fetch")
    probe = Path(tempfile.mkdtemp(prefix="frontier-probe-", dir=tmp))
    try:
        rc, out = sh(["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout", url, str(probe / "r")],
                     timeout=180)
        if rc != 0:
            return False, f"git clone {url} failed (exit {rc}): {out.strip()[:200]}"
        rc, _ = sh(["git", "-C", str(probe / "r"), "cat-file", "-e", commit + "^{commit}"])
        return rc == 0, (f"{url} reachable at {commit}" if rc == 0 else f"{url} has no commit {commit}")
    finally:
        shutil.rmtree(probe, ignore_errors=True)


def p10_disk(path: Path, min_gb: float) -> tuple[bool, str]:
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    free = shutil.disk_usage(p).free / 1e9
    return free > min_gb, f"{free:.0f} GB free under {p}" + ("" if free > min_gb else f", need > {min_gb:g} GB")


def p11_setup(session: Path) -> tuple[bool, str]:
    f = session / "PREFLIGHT.txt"
    if not f.is_file():
        return False, f"no {f} (run setup_vm.sh first)"
    v = [x for x in f.read_text().splitlines() if x.startswith("verdict ")]
    if v and v[-1].startswith("verdict READY"):
        return True, f"{f}: {v[-1]}"
    return False, f"{f}: {v[-1] if v else 'no verdict line'}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    moe = Path(os.environ.get("MOE_HOME", str(Path.home() / "moe")))
    ap.add_argument("--repo", default=str(HERE.parents[1]))
    ap.add_argument("--model", required=True, help="a prodlib.MODELS key (the blind registry included)")
    ap.add_argument("--expect-commit", default=None)
    ap.add_argument("--power-w", type=int, default=700)
    ap.add_argument("--locks", default="1710,1410")
    ap.add_argument("--tmpdir", default=os.environ.get("TMPDIR") or str(moe / "tmp"))
    ap.add_argument("--nsys", default="nsys")
    ap.add_argument("--session", default=os.environ.get("SESSION_ROOT") or str(moe / "session"))
    ap.add_argument("--frontier-setup", default=str(HERE / "frontier_setup.sh"))
    ap.add_argument("--frontier-url", default=None, help="default: frontier_setup.sh's URL")
    ap.add_argument("--frontier-commit", default=None, help="default: frontier_setup.sh's COMMIT")
    ap.add_argument("--frontier-src", default=str(moe / "frontier" / "src"))
    ap.add_argument("--disk", default=str(moe))
    ap.add_argument("--min-free-gb", type=float, default=200.0)
    ap.add_argument("--rate-url", default=None, help="P12: the URL to time (default: the first weight shard at the pin)")
    ap.add_argument("--rate-bytes", type=float, default=1e9)
    ap.add_argument("--rate-seconds", type=float, default=20.0)
    ap.add_argument("--min-mbps", type=float, default=50.0)
    ap.add_argument("--skip", default="", help="comma list of check ids not run (tests only)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    tmp = Path(a.tmpdir)
    skip = {x for x in a.skip.split(",") if x}
    expect, src = expected_commit(a.expect_commit, moe)
    furl, fcommit = frontier_pin(Path(a.frontier_setup))
    furl, fcommit = a.frontier_url or furl, a.frontier_commit or fcommit
    checks = [
        ("P1", "git", lambda: p1_git(Path(a.repo), expect, src)),
        ("P2", "idle", p2_idle),
        ("P3", "power", lambda: p3_power(a.power_w)),
        ("P4", "locks", lambda: p4_locks([int(x) for x in a.locks.split(",") if x])),
        ("P5", "tmpdir", lambda: p5_tmpdir(tmp)),
        ("P6", "tools", lambda: p6_tools(a.nsys)),
        ("P7", "nsys", lambda: p7_nsys(a.nsys, tmp)),
        ("P8", "hf-rev", lambda: p8_hf(a.model)),
        ("P9", "frontier", lambda: p9_frontier(furl, fcommit, Path(a.frontier_src), tmp)
         if furl and fcommit else (False, f"no URL/COMMIT in {a.frontier_setup}")),
        ("P10", "disk", lambda: p10_disk(Path(a.disk), a.min_free_gb)),
        ("P11", "setup", lambda: p11_setup(Path(a.session))),
        ("P12", "hf-rate", lambda: p12_rate(a.model, a.rate_url, a.rate_bytes, a.rate_seconds, a.min_mbps)),
    ]
    res = []
    for cid, what, fn in checks:
        if cid in skip:
            res.append({"id": cid, "what": what, "verdict": "SKIP", "detail": "--skip"})
            continue
        if cid == "P4" and res[1]["verdict"] == "FAIL":
            res.append({"id": cid, "what": what, "verdict": "FAIL", "detail": "not tried: the GPU is busy (P2)"})
            continue
        try:
            ok, detail = fn()
        except Exception as exc:                                      # noqa: BLE001
            ok, detail = False, f"raised {type(exc).__name__}: {exc}"
        res.append({"id": cid, "what": what, "verdict": "PASS" if ok else "FAIL", "detail": detail})
    for r in res:
        print(f"{r['verdict']} {r['id']} {r['what']}: {r['detail']}")
    bad = [r for r in res if r["verdict"] == "FAIL"]
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"schema": "moe-kernels/prod-env-preflight/1", "model": a.model,
                                           "checks": res, "pass": not bad}, indent=2) + "\n")
    if bad:
        print("FAIL env-preflight: " + "; ".join(f"{r['id']} {r['what']}" for r in bad))
        return 3
    print("PASS env-preflight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
