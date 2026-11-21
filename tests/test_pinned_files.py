"""Every file a registration pins stays byte-identical through the tile stage (the build brief's
standing rule): R3 (scripts/private_weight_reference.py), scripts/dram_counter_route.py at rental 7's
sha 130d5dc..., and every earlier registration's code pins. New behaviour lives in new modules
(scripts/r3_tile.py, dram_counter_route_tile.py, r3_timing_model_v4.py, wave_split_bytes_v4.py,
tile_census.py, skew_synth_tile.py, gh200_tile_session.sh and scripts/scoring/tileA/).

Rentals 3 to 5's pins were already superseded before this stage (their files moved at rentals 5 and
6, by registration); BASELINE holds every pinned file's sha256 at the tile stage's starting commit
(80514a6), so no pinned file may move here whatever its registration's date, and the pins of
rentals 6 and 7 (the ones in force) must equal their registered sha256 exactly."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REG = REPO / "docs" / "registered"

#: sha256 at 80514a6 of every file any registration before the tile stage pins
BASELINE = {
    "moe/instrumented/__init__.py": "331d8130c6aaee92c0e0cedab7fa7d2f79fc1479c1a5ddfea8cd1b3b44141488",
    "moe/instrumented/cubin.py": "3a252162ba5af0da7e71ed86f80e940ba4cd3035e16503c4e48569251a66a28f",
    "moe/instrumented/fused_moe_instr.py": "16a8e39877add06bdab08de7de53de0808a0171a468816cb3c9ee23594594250",
    "scripts/cross_model_predict.py": "a8a354326121972b53cbed03922177dc9c4a00805fc817b939142e9aa49fb08d",
    "scripts/cross_model_score.py": "d770e8ba1c4178add7c7180921d434aca8c4ee115068809cb5f0227dad22ef07",
    "scripts/dram_counter_route.py": "130d5dc894d5355080b193d3e63468672e3f756c457df26a65c25b9216630e3c",
    "scripts/floor_estimator.py": "40c785ea144740ae663ed11f5692b0978f2436fa836bf682fd5cc71201989cda",
    "scripts/gh200_model_session.sh": "901c18a290990ea86d7fc604d97ae2210a17a980bfedb32de845d2e9b47e5787",
    "scripts/gpubench.py": "591def14403dbaf4e7023ec176f95980a41e93d047226e817604a7da162a1a6c",
    "scripts/hw_rulers.py": "8615e321ba45ab34ec8a4d7ce650e07a523969f684b3b2d971ca910fc5994415",
    "scripts/instr_probe.py": "a5b421bfd6825da67a8241a9535319c83824620928309633ac3b096ca479a6fe",
    "scripts/l2_survival.py": "cd85277d931ec7d09d27a02acfdf5c887142571b2456825b1bf809ae6a56e09b",
    "scripts/lock_gate.py": "71c56a15a2ea168b5669a36363e941aa310acdbe51dd794a0a2bc6f7bd26f1d4",
    "scripts/locked_r3.py": "6e44670538fa3d2e6a158295909f71a41fdb5eeac507798294bea6d436cb5aa6",
    "scripts/private_weight_reference.py": "daa4070636866c9cf695d1fc7133a2077cb518728badc9a31b352406d0a4f70d",
    "scripts/r3_timing_model.py": "09ef7543ffe53e733f1eef5dd8b5150b7f806cfbb3db15da6ad7537f73811ef0",
    "scripts/r3_timing_model_v3.py": "595982d3dc0b4b6eea919cbdbbc674150d11d00c39fdd4d08deb6f826e2e8c07",
    "scripts/scoring/rental3/r3common.py": "5dfdb3e2d8fa85634e0577df389e514659f9ff337d03ce7c3a28f105c669f305",
    "scripts/scoring/rental4/r4common.py": "f2cb03333cce44470855b0633292af135084b755352e10e0334d09a94ff37665",
    "scripts/scoring/rental5/r5common.py": "b3ae6f0d7feaa733e5c1aa7ed2f34706025bc02739aac6f4536ed8d3567e1086",
    "scripts/scoring/rental5/skewmodel.py": "262b785e976dd6e712288c69e67dae7b429231f2c1adb4c389205b37f51e0b11",
    "scripts/scoring/rental6/r6common.py": "59b096345ce7e00f56cdc76fe7d37226c37a49651f04d54811a614c343ea06b9",
    "scripts/scoring/rental6/r6model.py": "04c2042561085201a1b4c679be07db007369a3bdf71cefdd746f23ef488c045b",
    "scripts/scoring/rental6/reg6.py": "d3632633b1009362087035300f55a7fd76b69b2b8afac1a0340d7897928cff03",
    "scripts/scoring/rental6/rowkeys.py": "7a5fa23be6d2dc39542ac8555f1a06bd5fad5f9d3a65518c6e6a369c240e7633",
    "scripts/scoring/rental7/r7common.py": "6e224a8e70944ea6c7127b724a286d861030a4a4c0b67cb846bbe7f66f1d9851",
    "scripts/scoring/rental7/reg7.py": "392af92b6117e02587911eab9d0023ffe4cd5c37f5d21f78d1ddef95d3fa5150",
    "scripts/scoring/rental7/rowkeys7.py": "cfca6adee49009f4d6941d73f0dda5f058fe94c101d7c332439e35b799fa639b",
    "scripts/skew_synth.py": "228e69bef533a4654b94d851cf7472eaa503eb901eac084a51dc8a5595316f03",
    "scripts/skew_synth7.py": "e241296eb4d341ba05c7c55e1ba9e97898cf8339fbe074968c367de35ead6a0a",
    "scripts/wave_split_bytes.py": "3693e0a1c34624ef928ff43876418009adb4341496fc5821ce2bd1ec8c952b39",
    "scripts/wave_split_bytes_v3.py": "24992697b2ba19e13be0df64dba200498c6796ebc6517776af85253df266a066",
}


def _sha(p: str) -> str:
    return hashlib.sha256((REPO / p).read_bytes()).hexdigest()


def _pins(glob: str) -> dict:
    out = {}
    for f in sorted(REG.glob(glob)):
        for p, s in (json.loads(f.read_text()).get("code_pins") or {}).items():
            out.setdefault(p, set()).add(s)
    return out


@pytest.mark.parametrize("path", sorted(BASELINE))
def test_every_pinned_file_is_byte_identical_to_the_stage_baseline(path):
    assert _sha(path) == BASELINE[path], path


def test_r3_and_the_counter_route_are_rental_7s_bytes():
    assert _sha("scripts/dram_counter_route.py").startswith("130d5dc")
    assert _sha("scripts/private_weight_reference.py").startswith("daa40706")


@pytest.mark.parametrize("glob", ["*-rental6-*.json", "*-rental7-*.json", "*-tileA-*.json", "*-tileB-*.json"])
def test_the_pins_in_force_equal_their_registered_sha256(glob):
    pins = _pins(glob)
    assert pins, glob
    for p, shas in pins.items():
        assert len(shas) == 1, (p, shas)
        assert _sha(p) == next(iter(shas)), p


def test_every_file_rental_7_pins_is_pinned_again_by_the_tile_stage():
    r7, tile = _pins("*-rental7-*.json"), _pins("*-tileA-*.json")
    assert set(r7) <= set(tile)
    for p in r7:
        assert r7[p] == tile[p], p
