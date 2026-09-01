# vLLM tuned fused-MoE configs, snapshotted at `v0.27.1`

These JSON files are vendored verbatim from vLLM so that
`moe/bench/tile_resolve.py` can resolve a tile OFFLINE and reproducibly. Nothing
in this directory is a measurement. It is upstream's grid-search output, and a
tile derived from it is DERIVED, never observed -- see the module docstring for
why that distinction is load-bearing here.

    upstream tag   v0.27.1
    upstream path  vllm/model_executor/layers/fused_moe/configs/
    fetched on     2026-09-01
    fetched from   https://raw.githubusercontent.com/vllm-project/vllm/v0.27.1/vllm/model_executor/layers/fused_moe/configs/<name>

## What is here, and why only four files

327 config files ship at that tag. Four of them are the ones this study's cells
can reach, so four are vendored. The rest would be dead weight in a repository
whose point is that every committed number can be checked.

| file | sha256 | which cells |
|---|---|---|
| `E=8,N=14336,device_name=NVIDIA_H200.json` | `082fe3b0c857ceb6277bb7883aba036b47c204a3677df89a556d9329de0ff9e9` | mixtral-8x7b, H200, bf16 |
| `E=64,N=2560,device_name=NVIDIA_H200.json` | `097887946a5e6b97041a6162dbc25e737e2ec446484c5d9795d545709b5e657d` | qwen2-57b-a14b, H200, bf16 |
| `E=8,N=14336,device_name=NVIDIA_H200,dtype=fp8_w8a8.json` | `41a213148254e14b2b3815da19d3d82704e6609d274fd9228aa794899cef7759` | mixtral-8x7b, H200, fp8 |
| `E=64,N=2560,device_name=NVIDIA_H200,dtype=fp8_w8a8.json` | `758a4a82fb86ec98e2d16d3383f13d35f2288d2b4101d05c29de01de78629aa9` | qwen2-57b-a14b, H200, fp8 |

Those are 4 of the 16 `(model x card x dtype)` combinations this study measured.
The other 12 ship no file at all: nothing exists for `NVIDIA_A100-SXM4-80GB` at
any of the four shapes, and nothing exists for deepseek-v3's unsharded
`E=256,N=2048` or deepseek-v2-lite's `E=64,N=1408` on any device. Those cells
took `get_default_config`'s hardcoded ladder, which `tile_resolve.default_config`
reimplements.

## `SHIPPED_FILE_NAMES.txt` -- and why the whole listing is here when the files are not

All 327 names, one per line, sorted. It is what makes a NEGATIVE answer
available: "this file ships" needs one name, "NO file ships for this shape, so
the cell took the fallback ladder" needs all of them.

Without it, a shape whose file was simply never vendored and a shape upstream
never tuned look identical -- both are "not on disk" -- and the resolver would
answer `vllm_default_derived` for both. One of those two answers would be wrong
in the direction this whole batch of work exists to prevent: a plausible tile
with nothing behind it. With the listing, a name that ships but is absent from
this directory raises `SnapshotMissing` and prints the command to add it.

`tests/test_tile_resolve.py` pins this listing equal to the independent
transcription in `tests/test_deployment_shapes.py`, so a slip in either is loud.

## Adding a shape

Two steps, and the first alone is not enough:

    cd moe/bench/hardware/vllm_configs
    name='E=128,N=768,device_name=NVIDIA_H200.json'
    curl -sSf -o "$name" \
      "https://raw.githubusercontent.com/vllm-project/vllm/v0.27.1/vllm/model_executor/layers/fused_moe/configs/$(python3 -c 'import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))' "$name")"
    shasum -a 256 "$name"     # then add the row to the table above

If the name is not already in `SHIPPED_FILE_NAMES.txt`, it is not a v0.27.1
file, and adding it here would make the resolver claim a tuned config that the
measured run could not have loaded.

## Regenerating `SHIPPED_FILE_NAMES.txt`

    curl -sS "https://api.github.com/repos/vllm-project/vllm/contents/vllm/model_executor/layers/fused_moe/configs?ref=v0.27.1&per_page=100" \
      | python3 -c "import json,sys; print('\n'.join(sorted(e['name'] for e in json.load(sys.stdin))))"

## What this snapshot cannot tell you

- `VLLM_TUNED_CONFIG_FOLDER` is consulted BEFORE the shipped directory, so a
  user-supplied JSON can tune any shape. No sweep in this study sets it.
- `VLLM_BATCH_INVARIANT` makes `get_moe_configs` return `None` before any lookup
  and pins the config at `BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
  GROUP_SIZE_M=8`. No sweep in this study sets it either.
- A vLLM that is not v0.27.1 ships a different tree. The version is not in the
  schema, so that scope is carried by assertion rather than by data.
