"""Tile efficiency at any BLOCK_M, from a published row's own columns.

The schema stores `load_tile_eff_bm64` and `load_tile_eff_bm128`, and neither is
the tile the Triton baselines actually run: vLLM's tuned H200 config sets
`BLOCK_SIZE_M` to 16 for every batch size from 1 to 256, which is the whole
decode range. So every stored tile-efficiency figure describes a configuration
vLLM does not use at decode.

Two fixes were rejected before this one. Adding a `bm16` column fails because
`schema.read_csv` refuses an unrecognised schema_version, so it would make all
17,640 published rows unreadable. Regenerating the routing fails because
`cli.build_routing_source` passes `device=args.device`, so a GPU run samples with
a CUDA generator, and CUDA and CPU RNG differ for the same seed. Published
routing is not reproducible off the GPU.

What works is arithmetic on columns the row already carries, valid exactly while
every expert fits one tile. Outside that range it raises instead of guessing.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from moe.routing.imbalance import TileEfficiencyUndetermined, tile_efficiency_for_row

PUBLISHED = (Path(__file__).resolve().parents[1] / "results" / "published"
             / "2026-08-26-nvidia_h200-full-three-way" / "merged.csv")


def published_rows(limit: int = 3000) -> list[dict]:
    if not PUBLISHED.exists():
        pytest.skip(f"no published sweep at {PUBLISHED}")
    with PUBLISHED.open(newline="") as fh:
        return [r for _, r in zip(range(limit), csv.DictReader(fh), strict=False)]


def usable(row: dict) -> bool:
    try:
        return (int(float(row["load_active_experts"])) > 0
                and float(row["load_total_rows"]) > 0)
    except (KeyError, ValueError):
        return False


@pytest.mark.parametrize("block_m,column", [(64, "load_tile_eff_bm64"),
                                            (128, "load_tile_eff_bm128")])
def test_it_reproduces_the_column_the_harness_wrote(block_m, column):
    """The only honest check: match what the harness recorded while it held the
    real topk_ids, on every row where the identity is supposed to hold."""
    checked = 0
    for row in published_rows():
        if not usable(row) or float(row["load_max_rows"]) > block_m:
            continue
        stored = float(row[column])
        if stored == 0.0:
            continue
        assert tile_efficiency_for_row(row, block_m) == pytest.approx(stored, rel=1e-9), (
            f"{row['model']}/T{row['num_tokens']}/{row['routing_kind']} at bm={block_m}")
        checked += 1
    assert checked > 500, f"only {checked} rows exercised"


def test_it_answers_for_the_tile_vllm_actually_uses():
    """bm16 has no stored column to check against, so assert the properties any
    tile efficiency must have, plus monotonicity against bm64."""
    checked = 0
    for row in published_rows():
        if not usable(row) or float(row["load_max_rows"]) > 16:
            continue
        eff16 = tile_efficiency_for_row(row, 16)
        assert 0.0 < eff16 <= 1.0
        # ceil(n/m)*m is non-decreasing in m, so a smaller tile never wastes more.
        assert eff16 >= tile_efficiency_for_row(row, 64) - 1e-12
        checked += 1
    assert checked > 100, f"only {checked} rows exercised"


def test_a_hot_expert_spanning_tiles_is_refused_not_guessed():
    """Above the threshold the answer needs the full distribution, which is not
    stored and cannot be regenerated off-GPU. Silence would be the bug."""
    spanning = [r for r in published_rows()
                if usable(r) and float(r["load_max_rows"]) > 16]
    assert spanning, "no rows where an expert spans more than one 16-row tile"
    with pytest.raises(TileEfficiencyUndetermined):
        tile_efficiency_for_row(spanning[0], 16)


def test_a_tile_of_one_row_wastes_nothing():
    for row in published_rows(200):
        if usable(row) and float(row["load_max_rows"]) <= 1:
            assert tile_efficiency_for_row(row, 1) == pytest.approx(1.0)


def test_a_nonsense_block_m_is_rejected():
    row = next(r for r in published_rows(200) if usable(r))
    with pytest.raises(ValueError):
        tile_efficiency_for_row(row, 0)
