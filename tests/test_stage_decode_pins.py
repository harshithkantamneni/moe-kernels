"""Every pinned file stays byte-identical through the decode stage (the build brief's standing rule): R3,
dram_counter_route.py, gh200_model_session.sh, gh200_tile_session.sh, every tile-stage module and every
earlier registration's code pins, at their sha256 of the decode stage's starting commit (eb84555, the
tile plumbing included). New behaviour lives in new modules only (scripts/r3_decode.py, decode_synth.py,
r3_timing_model_v4d.py, dram_counter_route_decode.py, gh200_decode_session.sh, scripts/scoring/decode/).
The decode registrations' own pins must equal their files."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REG = REPO / "docs" / "registered"

#: sha256 at eb84555 of every file the tile stage's registrations and tests/test_pinned_files.py pin
BASELINE = {
    "moe/instrumented/__init__.py": "331d8130c6aaee92c0e0cedab7fa7d2f79fc1479c1a5ddfea8cd1b3b44141488",
    "moe/instrumented/cubin.py": "3a252162ba5af0da7e71ed86f80e940ba4cd3035e16503c4e48569251a66a28f",
    "moe/instrumented/fused_moe_instr.py": "16a8e39877add06bdab08de7de53de0808a0171a468816cb3c9ee23594594250",
    "scripts/cross_model_predict.py": "a8a354326121972b53cbed03922177dc9c4a00805fc817b939142e9aa49fb08d",
    "scripts/cross_model_score.py": "d770e8ba1c4178add7c7180921d434aca8c4ee115068809cb5f0227dad22ef07",
    "scripts/dram_counter_route.py": "130d5dc894d5355080b193d3e63468672e3f756c457df26a65c25b9216630e3c",
    "scripts/dram_counter_route_tile.py": "e180be82a3330e8c10aa0da4d41eb2759fee2a21e5346222fb8f824b94345ecc",
    "scripts/floor_estimator.py": "40c785ea144740ae663ed11f5692b0978f2436fa836bf682fd5cc71201989cda",
    "scripts/gh200_model_session.sh": "901c18a290990ea86d7fc604d97ae2210a17a980bfedb32de845d2e9b47e5787",
    "scripts/gh200_tile_session.sh": "0f7308e97b7755ea5187940b5f6a13cee39b4740040443879f08fb2aba1507e9",
    "scripts/gpubench.py": "591def14403dbaf4e7023ec176f95980a41e93d047226e817604a7da162a1a6c",
    "scripts/hw_rulers.py": "8615e321ba45ab34ec8a4d7ce650e07a523969f684b3b2d971ca910fc5994415",
    "scripts/instr_probe.py": "a5b421bfd6825da67a8241a9535319c83824620928309633ac3b096ca479a6fe",
    "scripts/l2_survival.py": "cd85277d931ec7d09d27a02acfdf5c887142571b2456825b1bf809ae6a56e09b",
    "scripts/lock_gate.py": "71c56a15a2ea168b5669a36363e941aa310acdbe51dd794a0a2bc6f7bd26f1d4",
    "scripts/locked_r3.py": "6e44670538fa3d2e6a158295909f71a41fdb5eeac507798294bea6d436cb5aa6",
    "scripts/private_weight_reference.py": "daa4070636866c9cf695d1fc7133a2077cb518728badc9a31b352406d0a4f70d",
    "scripts/r3_tile.py": "be9ee1991a9765881ca82e92665b4538b39295a98e7445b7828d5b1cb3cbf1cf",
    "scripts/r3_timing_model.py": "09ef7543ffe53e733f1eef5dd8b5150b7f806cfbb3db15da6ad7537f73811ef0",
    "scripts/r3_timing_model_v3.py": "595982d3dc0b4b6eea919cbdbbc674150d11d00c39fdd4d08deb6f826e2e8c07",
    "scripts/r3_timing_model_v4.py": "e3c2b4143cd5751785140381fa61b4fa39fdbea65762fa836c48b6ab54887c14",
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
    "scripts/scoring/tileA/finalise.py": "a463577e7ec55b44483ae94c95fe1c0dd01b281443240d5d4766aedd0442bfe0",
    "scripts/scoring/tileA/power_tile.py": "9b5804cfe351108ff451f1a058997ef426daf5f5f90b5b522c5016dc3926c9d5",
    "scripts/scoring/tileA/regtile.py": "ee331a0b62ebae5a209d689ce3f3e33aaa7f480aa5c0d8e342a0824c5d942686",
    "scripts/scoring/tileA/rowkeys_tile.py": "0575d71bca4b031147daa76a0f75e16d80ce051f02690c9c2f7459417cb18172",
    "scripts/scoring/tileA/tcommon.py": "1b94cc123ce6169e3c49482e6bf3b975a3b6cf1e4f0f8cad4739676ec6e68d6d",
    "scripts/skew_synth.py": "228e69bef533a4654b94d851cf7472eaa503eb901eac084a51dc8a5595316f03",
    "scripts/skew_synth7.py": "e241296eb4d341ba05c7c55e1ba9e97898cf8339fbe074968c367de35ead6a0a",
    "scripts/skew_synth_tile.py": "f010c8defe1a96f17465000cbfe5075d02c2a03d3be5816665a4a4ac7d03150e",
    "scripts/tile_census.py": "8dbdb8439b59f621ebe2b6a3e95e2c43b2aa01aeeae965eed0053a5e5786045e",
    "scripts/wave_split_bytes.py": "3693e0a1c34624ef928ff43876418009adb4341496fc5821ce2bd1ec8c952b39",
    "scripts/wave_split_bytes_v3.py": "24992697b2ba19e13be0df64dba200498c6796ebc6517776af85253df266a066",
    "scripts/wave_split_bytes_v4.py": "927f9c2df7d3fb70e546dcd632a901cd4e414f4e65f959ba6a20da2cc4332265",
}


def _sha(p: str) -> str:
    return hashlib.sha256((REPO / p).read_bytes()).hexdigest()


@pytest.mark.parametrize("path", sorted(BASELINE))
def test_every_pinned_file_is_byte_identical_to_the_decode_stage_baseline(path):
    assert _sha(path) == BASELINE[path], path


def test_the_baseline_covers_every_tile_stage_pin_and_every_earlier_baseline():
    import re
    tile = set()
    for f in REG.glob("*-tile[AB]-*.json"):
        tile |= set(json.loads(f.read_text()).get("code_pins") or {})
    old = set(re.findall(r'^    "([^"]+)": "[0-9a-f]{64}",$', (REPO / "tests" / "test_pinned_files.py").read_text(), re.M))
    assert tile | old <= set(BASELINE)
    for p in ("scripts/private_weight_reference.py", "scripts/dram_counter_route.py", "scripts/gh200_model_session.sh",
              "scripts/gh200_tile_session.sh", "scripts/locked_r3.py"):
        assert p in BASELINE


@pytest.mark.parametrize("name", ["2026-10-14-decodeC-cal-gh200", "2026-10-14-decodeB-blind-gh200"])
def test_the_decode_registrations_pin_their_files_at_their_bytes(name):
    pins = json.loads((REG / f"{name}.json").read_text())["code_pins"]
    for p, s in pins.items():
        assert _sha(p) == s, p
    for p in BASELINE:
        if p in pins:
            assert pins[p] == BASELINE[p], p
    assert "scripts/r3_decode.py" in pins and "scripts/gh200_decode_session.sh" in pins


def test_no_decode_module_shadows_an_earlier_scorer_by_name():
    mine = {p.name for p in (REPO / "scripts" / "scoring" / "decode").glob("*.py")}
    others = {p.name for d in (REPO / "scripts" / "scoring").iterdir() if d.is_dir() and d.name != "decode"
              for p in d.glob("*.py")}
    assert not mine & others, mine & others


def test_models_v4_and_v4d_share_the_one_registered_r3_timing_model():
    """tests/test_r3_timing_model.py reuses an already-imported r3_timing_model instead of replacing
    it, so v4 and v4d bind the registered module whatever order the test files collect in."""
    sys.path.insert(0, str(REPO / "scripts"))
    import r3_timing_model_v4 as V4
    import r3_timing_model_v4d as V4D
    spec = importlib.util.spec_from_file_location("_t_r3tm", REPO / "tests" / "test_r3_timing_model.py")
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    assert T.M is sys.modules["r3_timing_model"] is V4.TM is V4D.TM
