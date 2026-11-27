import tomllib
from pathlib import Path

import pytest
from packaging.markers import Marker


@pytest.mark.parametrize("python_version", ["3.13.7", "3.14.0"])
@pytest.mark.parametrize("platform,system,machine,extra,suffix", [
    ("darwin", "Darwin", "arm64", "", ""),
    ("darwin", "Darwin", "arm64", "cuda", ""),
    ("linux", "Linux", "x86_64", "", "+cpu"),
    ("linux", "Linux", "x86_64", "cuda", "+cu126"),
])
def test_lock_selects_one_matching_torch_pair(
    python_version, platform, system, machine, extra, suffix
):
    lock_path = Path(__file__).resolve().parents[1] / "poetry.lock"
    with lock_path.open("rb") as lock_file:
        packages = tomllib.load(lock_file)["package"]
    environment = {
        "python_full_version": python_version,
        "python_version": ".".join(python_version.split(".")[:2]),
        "sys_platform": platform,
        "platform_system": system,
        "platform_machine": machine,
        "os_name": "posix",
        "extra": extra,
    }
    selected = [
        (package["name"], package["version"])
        for package in packages
        if package["name"] in ("torch", "torchvision")
        and Marker(package.get("markers", "python_version >= '0'")).evaluate(environment)
    ]
    assert selected == [("torch", "2.10.0" + suffix), ("torchvision", "0.25.0" + suffix)]
