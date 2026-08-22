import pytest

import moe
from moe.spec import MODEL_CONFIGS, BenchSpec, RoutingSpec

moe.bootstrap("reference")


@pytest.fixture
def toy_spec():
    return BenchSpec(MODEL_CONFIGS["toy"], num_tokens=32, dtype="fp32",
                     routing=RoutingSpec("uniform"), seed=0)


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a CUDA device")
