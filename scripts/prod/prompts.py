"""The fixed prompt set the serving run feeds vLLM (KIT-SPEC 3): real text, so the router sees real routing.

    python scripts/prod/prompts.py fetch --source sharegpt --dest DIR [--allow-unpinned]
    python scripts/prod/prompts.py build --source sharegpt --data DIR/<file> --tokenizer <hf repo> \\
        --n 4096 --seed 0 --max-input 2048 --out-dist sharegpt --out prompts.jsonl

SOURCES. Primary: ShareGPT (the ShareGPT_V3 split vLLM's own serving benchmarks read): the first human turn of
each conversation with at least one human and one gpt turn. Fallback: a pinned public-domain text (Gutenberg),
cut into consecutive token windows. Each source is pinned by URL and sha256; a source whose sha256 is not pinned
here is REFUSED unless --allow-unpinned (the SMOKE rental only, which prints the sha256 to pin before the kit
freezes; main.sh never passes it).

DETERMINISM. The set is a function of (file bytes, seed, tokenizer, n, max_input, out_dist) only: eligible
conversations in file order, `random.Random(seed).sample` of their indices, the prompt's tokens truncated to
max_input, the output length drawn from out_dist by the same generator. The written file carries the sha256 of
the source and of its own body, so two runs that differ anywhere differ in the header.

OUTPUT LENGTHS (--out-dist): sharegpt (the token length of the conversation's first gpt reply, clipped to
[min_out, max_out]), uniform:A,B, fixed:N, lognormal:MU,SIGMA (clipped). serve_capture sets ignore_eos so every
request generates exactly its max_tokens.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import urllib.request
from pathlib import Path

SOURCES = {
    "sharegpt": {
        "url": "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/"
               "ShareGPT_V3_unfiltered_cleaned_split.json",
        "file": "ShareGPT_V3_unfiltered_cleaned_split.json",
        #: PINNED AT THE SMOKE RENTAL (README "to pin before freeze"); None refuses without --allow-unpinned
        "sha256": None,
        "kind": "sharegpt",
    },
    "gutenberg-1342": {
        "url": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
        "file": "pg1342.txt",
        "sha256": None,
        "kind": "text",
    },
}
SCHEMA = "moe-kernels/prod-prompts/1"
MIN_INPUT = 4


class PromptError(RuntimeError):
    pass


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_pin(source: str, path, allow_unpinned: bool) -> str:
    got = sha256_file(path)
    want = SOURCES[source]["sha256"]
    if want is None:
        if not allow_unpinned:
            raise PromptError(f"source {source} has no pinned sha256 (got {got}); pin it in prompts.py SOURCES "
                              "before the main rental (only the smoke run may pass --allow-unpinned)")
    elif got != want:
        raise PromptError(f"source {source}: sha256 {got} != pinned {want}")
    return got


def fetch(source: str, dest, allow_unpinned: bool = False) -> tuple[Path, str]:
    s = SOURCES[source]
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / s["file"]
    if not out.exists():
        tmp = out.with_suffix(out.suffix + ".part")
        urllib.request.urlretrieve(s["url"], tmp)                    # noqa: S310 (pinned https URL)
        tmp.rename(out)
    return out, check_pin(source, out, allow_unpinned)


# ------------------------------------------------------------------------------------------------------------
# output-length distributions
# ------------------------------------------------------------------------------------------------------------

def parse_out_dist(spec: str) -> tuple[str, tuple]:
    kind, _, rest = spec.partition(":")
    try:
        if kind == "sharegpt" and not rest:
            return kind, ()
        if kind == "fixed":
            n = int(rest)
            if n < 1:
                raise ValueError
            return kind, (n,)
        if kind == "uniform":
            a, b = (int(x) for x in rest.split(","))
            if not 1 <= a <= b:
                raise ValueError
            return kind, (a, b)
        if kind == "lognormal":
            mu, sigma = (float(x) for x in rest.split(","))
            return kind, (mu, sigma)
    except ValueError:
        pass
    raise PromptError(f"--out-dist {spec!r}: sharegpt | fixed:N | uniform:A,B | lognormal:MU,SIGMA")


def draw_out_len(dist: tuple[str, tuple], rng: random.Random, reply_len: int | None, lo: int, hi: int) -> int:
    kind, p = dist
    if kind == "sharegpt":
        v = reply_len if reply_len is not None else lo
    elif kind == "fixed":
        v = p[0]
    elif kind == "uniform":
        v = rng.randint(p[0], p[1])
    else:
        v = int(round(math.exp(rng.gauss(p[0], p[1]))))
    return max(lo, min(hi, int(v)))


# ------------------------------------------------------------------------------------------------------------
# building the set
# ------------------------------------------------------------------------------------------------------------

def sharegpt_pairs(doc) -> list[tuple[str, str, str]]:
    """(id, first human turn, first gpt reply after it) of every eligible conversation, in file order."""
    out = []
    for i, conv in enumerate(doc):
        turns = conv.get("conversations") or []
        human = next((j for j, t in enumerate(turns) if t.get("from") == "human" and (t.get("value") or "").strip()),
                     None)
        if human is None:
            continue
        reply = next((t.get("value") for t in turns[human + 1:] if t.get("from") == "gpt"), None)
        if reply is None:
            continue
        out.append((str(conv.get("id", i)), turns[human]["value"], reply))
    return out


def build(*, kind: str, data_path, encode, n: int, seed: int, max_input: int, out_dist: str,
          min_out: int = 4, max_out: int = 1024, source_sha256: str = "", tokenizer_name: str = "") -> dict:
    """The prompt set as {"header": {...}, "rows": [...]} (deterministic; `encode(text) -> list[int]`)."""
    dist = parse_out_dist(out_dist)
    rng = random.Random(int(seed))
    rows = []
    if kind == "sharegpt":
        doc = json.loads(Path(data_path).read_text(encoding="utf-8"))
        pairs = sharegpt_pairs(doc)
        if len(pairs) < n:
            raise PromptError(f"{len(pairs)} eligible conversations, {n} asked")
        order = rng.sample(range(len(pairs)), len(pairs))
        for idx in order:
            if len(rows) == n:
                break
            cid, prompt, reply = pairs[idx]
            toks = list(encode(prompt))[:max_input]
            if len(toks) < MIN_INPUT:
                continue
            reply_len = len(encode(reply)) if dist[0] == "sharegpt" else None
            rows.append({"i": len(rows), "src": cid, "prompt_token_ids": [int(t) for t in toks],
                         "input_len": len(toks), "max_tokens": draw_out_len(dist, rng, reply_len, min_out, max_out)})
    elif kind == "text":
        if dist[0] == "sharegpt":
            raise PromptError("a text source has no replies: pick fixed/uniform/lognormal")
        toks = list(encode(Path(data_path).read_text(encoding="utf-8")))
        pos = 0
        while len(rows) < n:
            L = rng.randint(MIN_INPUT * 8, max_input)
            if pos + L > len(toks):
                pos = 0
                if L > len(toks):
                    raise PromptError("the text is shorter than one prompt")
            rows.append({"i": len(rows), "src": f"tok{pos}", "prompt_token_ids": [int(t) for t in toks[pos:pos + L]],
                         "input_len": L, "max_tokens": draw_out_len(dist, rng, None, min_out, max_out)})
            pos += L
    else:
        raise PromptError(f"unknown source kind {kind}")
    if len(rows) < n:
        raise PromptError(f"only {len(rows)} prompts of >= {MIN_INPUT} tokens, {n} asked")
    body = "\n".join(json.dumps(r, sort_keys=True) for r in rows)
    header = {"schema": SCHEMA, "kind": kind, "source_sha256": source_sha256, "tokenizer": tokenizer_name,
              "n": n, "seed": seed, "max_input": max_input, "out_dist": out_dist, "min_out": min_out,
              "max_out": max_out, "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
              "input_tokens": sum(r["input_len"] for r in rows), "output_tokens": sum(r["max_tokens"] for r in rows)}
    return {"header": header, "rows": rows}


def write(promptset: dict, out) -> None:
    lines = [json.dumps(promptset["header"], sort_keys=True)] + [json.dumps(r, sort_keys=True)
                                                                   for r in promptset["rows"]]
    Path(out).write_text("\n".join(lines) + "\n")


def load(path) -> dict:
    lines = Path(path).read_text().splitlines()
    if not lines:
        raise PromptError(f"{path} is empty")
    header = json.loads(lines[0])
    if header.get("schema") != SCHEMA:
        raise PromptError(f"{path}: schema {header.get('schema')!r}")
    rows = [json.loads(x) for x in lines[1:] if x.strip()]
    body = "\n".join(json.dumps(r, sort_keys=True) for r in rows)
    if hashlib.sha256(body.encode()).hexdigest() != header["body_sha256"]:
        raise PromptError(f"{path}: body sha256 does not match its header (edited?)")
    return {"header": header, "rows": rows}


def hf_encoder(repo: str):
    """The tokenizer vLLM itself loads for the model (so prompt token ids are the ones the engine would make),
    through vLLM's own loader; the module moved between releases, so both homes are tried (VM only)."""
    try:
        from vllm.tokenizers import get_tokenizer
    except ImportError:
        from vllm.transformers_utils.tokenizer import get_tokenizer
    tok = get_tokenizer(repo, trust_remote_code=True)
    return lambda text: tok.encode(text, add_special_tokens=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--source", choices=sorted(SOURCES), default="sharegpt")
    f.add_argument("--dest", required=True)
    f.add_argument("--allow-unpinned", action="store_true")
    b = sub.add_parser("build")
    b.add_argument("--source", choices=sorted(SOURCES), default="sharegpt")
    b.add_argument("--data", required=True)
    b.add_argument("--tokenizer", required=True)
    b.add_argument("--n", type=int, default=4096)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--max-input", type=int, default=2048)
    b.add_argument("--out-dist", default="sharegpt")
    b.add_argument("--min-out", type=int, default=4)
    b.add_argument("--max-out", type=int, default=1024)
    b.add_argument("--allow-unpinned", action="store_true")
    b.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "fetch":
            p, sha = fetch(a.source, a.dest, a.allow_unpinned)
            print(f"PASS fetch {a.source}: {p} sha256 {sha}" + ("" if SOURCES[a.source]["sha256"] else
                                                               "  (UNPINNED: pin this sha256 before freeze)"))
            return 0
        sha = check_pin(a.source, a.data, a.allow_unpinned)
        ps = build(kind=SOURCES[a.source]["kind"], data_path=a.data, encode=hf_encoder(a.tokenizer), n=a.n,
                   seed=a.seed, max_input=a.max_input, out_dist=a.out_dist, min_out=a.min_out, max_out=a.max_out,
                   source_sha256=sha, tokenizer_name=a.tokenizer)
        write(ps, a.out)
        h = ps["header"]
        print(f"PASS build: {a.out} n={h['n']} input {h['input_tokens']} output {h['output_tokens']} tokens, "
              f"body sha256 {h['body_sha256'][:16]}")
        return 0
    except (PromptError, OSError, ValueError) as exc:
        print(f"FAIL prompts: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
