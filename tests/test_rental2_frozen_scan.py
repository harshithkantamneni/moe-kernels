"""rental2/register.py reads a FROZEN list of the published G=64 floor captures (CAL_PAGES_2BF6FE38), the 22 pages as they
stood at its registration commit 2bf6fe38: each exists, and each matches its bytes at that commit (sha256 committed
here, so the test needs no git history). A live scan of results/published would add later sessions' captures."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts", REPO / "scripts" / "scoring" / "rental2"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(name: str, path: Path):
    """Load rental 2's register.py under its own name: a bare `import register` picks up whichever rental's register
    module an earlier test imported first."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R2 = _load("rental2_register", REPO / "scripts" / "scoring" / "rental2" / "register.py")

#: sha256 of each frozen capture as committed at 2bf6fe38 (git show 2bf6fe38:results/published/<path>)
AT_2BF6FE38 = {
    "2026-09-27-nvidia_gh200_480gb-session/results/2026-09-27-nvidia_gh200_480gb-r3-counters/r3f-g64-lock1710.json":
        "8f11b86733ebce812b05109d1033f2d5c3b807bd07dbf3a3c4de0cf5f63d9427",
    "2026-09-27-nvidia_gh200_480gb-session/results/2026-09-27-nvidia_gh200_480gb-r3-counters/r3f-g64.json":
        "b427b096eba59359d92d5582a71663a63870fd2defc6a2816e7840d8834aef03",
    "2026-09-28-nvidia_gh200_480gb-8x22b-session/results/2026-09-28-nvidia_gh200_480gb-r3-counters/r3f-g64-lock1710.json":
        "03267bd87406e32bdca23d6161d29af3bd9e9ef6722e014c2898df773cf778da",
    "2026-09-28-nvidia_gh200_480gb-8x22b-session/results/2026-09-28-nvidia_gh200_480gb-r3-counters/r3f-g64.json":
        "b27f9bf4357384d5e24b95381cb035b6006d76522f45e6688fc87615bf94b159",
    "2026-09-28-nvidia_gh200_480gb-qwen2-57b-session/results/2026-09-28-nvidia_gh200_480gb-r3-counters/r3f-g64-lock1710.json":
        "3348b347e45efd92ee14e9617e9c3fd3e3d45c15eb56f3cfd24651bb4067f8ad",
    "2026-09-28-nvidia_gh200_480gb-qwen2-57b-session/results/2026-09-28-nvidia_gh200_480gb-r3-counters/r3f-g64.json":
        "2d18da7c166ec11ae2ba562dd70bb3929da04aa590c83a3622b7008d588ac1db",
    "2026-09-29-nvidia_gh200_480gb-jetmoe-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/r3f-g64-lock1710.json":
        "a73f86a388985ca778b6fcb1c06319e235b02b6c2b469669ac2b416244eed8a2",
    "2026-09-29-nvidia_gh200_480gb-jetmoe-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/r3f-g64.json":
        "58a5b6d2145bb063baadad529f8617f18ca67e16964d028fe128ff747e853869",
    "2026-09-29-nvidia_gh200_480gb-olmoe-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/r3f-g64-lock1710.json":
        "dd6735d1b5c12a2deeb84c6f684d6ced0d7c72c7c617c116ebe5e7012be7586d",
    "2026-09-29-nvidia_gh200_480gb-olmoe-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/r3f-g64.json":
        "d45ab92ace1b43ec5e89bc7a27bddd3a096bb53538727001b8a4096126f59e9f",
    "2026-09-29-nvidia_gh200_480gb-phi3.5-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/r3f-g64-lock1710.json":
        "75426f54bc373996c4dd134f196a3bd344b6fe493fa4f70db3c87f8e403bda67",
    "2026-09-29-nvidia_gh200_480gb-phi3.5-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/r3f-g64.json":
        "a81f6c72ab21ff1cea738d6f343dc9426e7baa26c9f45ec21398dcb7e3d67e59",
    "2026-09-29-nvidia_gh200_480gb-qwen1.5-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/r3f-g64-lock1710.json":
        "76109b14ac32dd76894ad9ebf4cbf4b112a80f42deb3f96002c552ab62b7ee27",
    "2026-09-29-nvidia_gh200_480gb-qwen1.5-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters/r3f-g64.json":
        "6b5e4dedc135631c289548c5adcb913613af6a7857fe01873db779b1e2a328dc",
    "2026-09-30-nvidia_gh200_480gb-granite-session/results/2026-09-30-nvidia_gh200_480gb-r3-counters/r3f-g64-lock1710.json":
        "2e4541f9cec47fc38c914fb14d29fb5aacf4ca8f0e36d2d52fa471e1612a42dc",
    "2026-09-30-nvidia_gh200_480gb-granite-session/results/2026-09-30-nvidia_gh200_480gb-r3-counters/r3f-g64.json":
        "b3fe73564cf2061a9824e63f6f817cde680170735133ee96597523183694b1cd",
    "2026-09-30-nvidia_gh200_480gb-jetmoe-floor-session/results/2026-09-30-nvidia_gh200_480gb-r3-counters/r3f-g64-lock1710.json":
        "e1c1a3a9c6706b8e44b4320c3a94902cd033e06354716531cbb041ad1f267bcd",
    "2026-09-30-nvidia_gh200_480gb-jetmoe-floor-session/results/2026-09-30-nvidia_gh200_480gb-r3-counters/r3f-g64.json":
        "37fa733accd4012b65afae70769b08ec74a94c9bb78bfa89093645311af4e47d",
    "2026-09-30-nvidia_gh200_480gb-mixtral8x7b-floor-session/results/2026-09-30-nvidia_gh200_480gb-r3-counters/r3f-g64-lock1710.json":
        "6ac242e99dfcac5ad31c9fb1ebdb477f8f19718636850ccb94b766863bccbf73",
    "2026-09-30-nvidia_gh200_480gb-mixtral8x7b-floor-session/results/2026-09-30-nvidia_gh200_480gb-r3-counters/r3f-g64.json":
        "722ee0a7975c9f39258dabf3dc13e4967cf02cc6aaf55307a2df89036e2192d7",
    "2026-10-01-nvidia_gh200_480gb-rental1-session/results/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp8-floor-r3-counters/r3f-g64-lock1710.json":
        "1b35ce33ca8f9258e586559be98fdfb0ea669c77eadb491f3b7071ae50fcc430",
    "2026-10-01-nvidia_gh200_480gb-rental1-session/results/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp8-floor-r3-counters/r3f-g64.json":
        "ec972b35872045c8c479a82fa08619028c4578fafb5f1b5ed9ab91218b29cc36",
}


def test_the_frozen_scan_is_the_22_captures_at_2bf6fe38_byte_for_byte():
    assert R2.CAL_PAGES_2BF6FE38 == tuple(sorted(AT_2BF6FE38)) and len(AT_2BF6FE38) == 22
    for p, h in AT_2BF6FE38.items():
        f = REPO / "results" / "published" / p
        assert f.is_file(), p
        assert hashlib.sha256(f.read_bytes()).hexdigest() == h, p
    assert all("unlocked" not in p for p in AT_2BF6FE38)


def test_the_scan_reads_no_later_capture():
    src = (REPO / "scripts" / "scoring" / "rental2" / "register.py").read_text()
    assert "PUB.rglob(" not in src
    live = sorted(str(f.relative_to(R2.PUB)) for f in R2.PUB.rglob("r3f-g64*.json") if "unlocked" not in f.stem)
    assert set(AT_2BF6FE38) < set(live)          # later sessions added captures; the registration reads none of them
