"""The Hugging Face commit sha of every repo the kit loads (review 2, MINOR 3), resolved ON THE LAPTOP before the kit
freezes and recorded by hand into prodlib.HF_REVISIONS:

    python scripts/prod/hf_revisions.py              # prints {model key: sha} and the tokenizer repos' shas
    python scripts/prod/hf_revisions.py --check      # exit 3 unless prodlib.HF_REVISIONS equals what the Hub says now

THE REPOS: each served model's hf_repo (prodlib.MODELS) and each tokenizer repo smoke.sh / main.sh pass prompts.py
(their TOK= line). A tokenizer repo must be one of the models' repos: prompts.py pins a tokenizer by the revision of the
model whose hf_repo it is (prompts.tokenizer_revision), so a tokenizer repo of its own would run unpinned; this
script refuses one.

THE SOURCE: the Hub's public model API, GET https://huggingface.co/api/models/<repo> (its "sha" field: the commit
the default branch points at). No token, no cookie, no header beyond a generic User-Agent; nothing about the user
is sent. The VM never calls this: it loads the pinned sha (serve_capture G-REV fails a pinned run whose loaded
snapshot is not the pin, or whose snapshot cannot be resolved).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import prodlib as PL  # noqa: E402

API = "https://huggingface.co/api/models/{repo}"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
KIT_SCRIPTS = ("smoke.sh", "main.sh", "extras.sh")


def tokenizer_repos() -> list[str]:
    """The TOK= (and extras.sh's QTOK=) repos of smoke.sh, main.sh and extras.sh, in file order."""
    out = []
    for name in KIT_SCRIPTS:
        for m in re.finditer(r"^Q?TOK=(\S+)\s*$", (HERE / name).read_text(), re.M):
            out.append(m.group(1).strip("'\""))
    return out


def fetch_sha(repo: str, *, timeout: float = 20.0) -> str:
    req = urllib.request.Request(API.format(repo=repo), headers={"User-Agent": "moe-kernels-prod-kit/1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:      # noqa: S310 (a fixed https URL)
        doc = json.loads(resp.read().decode())
    return str(doc.get("sha") or "")


def resolve(fetch=fetch_sha) -> dict:
    """{"models": {key: sha}, "tokenizers": {repo: sha}}; PL.KitError on a tokenizer repo that is no model's repo or
    on an answer that is not a 40-hex commit."""
    repos = {m.key: m.hf_repo for m in PL.MODELS.values()}
    toks = tokenizer_repos()
    stray = sorted(set(toks) - set(repos.values()))
    if stray:
        raise PL.KitError(f"tokenizer repo(s) {stray} are no served model's hf_repo: prompts.py could not pin them")
    shas = {}
    for repo in sorted(set(repos.values()) | set(toks)):
        sha = fetch(repo)
        if not SHA_RE.match(sha):
            raise PL.KitError(f"{repo}: the Hub answered sha {sha!r}, not a 40-hex commit")
        shas[repo] = sha
    return {"models": {k: shas[r] for k, r in sorted(repos.items())}, "tokenizers": {t: shas[t] for t in toks}}


def check(got: dict, pinned: dict) -> list[str]:
    """Differences between the Hub's shas and prodlib.HF_REVISIONS (empty: the pins are current)."""
    return [f"{k}: pinned {pinned.get(k)!r}, the Hub says {v}" for k, v in sorted(got["models"].items())
            if pinned.get(k) != v]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 3 unless prodlib.HF_REVISIONS matches the Hub")
    a = ap.parse_args(argv)
    try:
        got = resolve()
    except (PL.KitError, OSError, ValueError) as exc:
        print(f"FAIL hf_revisions: {type(exc).__name__}: {exc}")
        return 3
    print(json.dumps(got, indent=1))
    print("HF_REVISIONS: dict = " + json.dumps(got["models"]))
    if a.check:
        bad = check(got, PL.HF_REVISIONS)
        if bad:
            print("FAIL hf_revisions --check: " + "; ".join(bad))
            return 3
        print("PASS hf_revisions --check: prodlib.HF_REVISIONS is the Hub's current commit of every repo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
