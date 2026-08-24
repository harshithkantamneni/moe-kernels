"""Build flags derive from the attached device, not from a hardcoded Hopper.

`docs/KERNEL_INTEGRATION.md` hardcoded `-gencode=arch=compute_90a,code=sm_90a`,
which is right for H100/H200 and silently wrong everywhere else: an A100 is
sm_80, RTX Blackwell is sm_120, B200 is sm_100. Building for the wrong arch
either fails to compile or loses the instructions the architecture exists for.
"""
import pytest

from moe.kernels import cuda_arch_flags


@pytest.mark.parametrize("capability,expected", [
    ((9, 0), "-gencode=arch=compute_90a,code=sm_90a"),     # H100 / H200
    ((8, 0), "-gencode=arch=compute_80,code=sm_80"),       # A100
    ((8, 9), "-gencode=arch=compute_89,code=sm_89"),       # L40S / RTX Ada
    ((10, 0), "-gencode=arch=compute_100a,code=sm_100a"),  # B200
    ((12, 0), "-gencode=arch=compute_120a,code=sm_120a"),  # RTX PRO 6000
])
def test_arch_flag_follows_the_device_capability(capability, expected):
    assert cuda_arch_flags(capability) == [expected]


def test_hopper_and_later_get_the_a_variant():
    """The `a` suffix is what exposes wgmma and TMA. Without it you compile for
    the architecture but lose the instructions it exists for."""
    assert "sm_90a" in cuda_arch_flags((9, 0))[0]
    assert "sm_100a" in cuda_arch_flags((10, 0))[0]


def test_pre_hopper_has_no_a_variant():
    """sm_80a does not exist; emitting it would fail the nvcc invocation."""
    assert cuda_arch_flags((8, 0)) == ["-gencode=arch=compute_80,code=sm_80"]


def test_unknown_capability_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="capability"):
        cuda_arch_flags((0, 0))
