from logging import info, basicConfig, INFO
from subprocess import check_output
from typing import List

import pkg_resources as pkg


def ensure_installed(requirements: List[str]):
    for requirement in requirements:
        try:
            pkg.require(requirement)
        except:
            basicConfig(format="%(message)s", level=INFO)
            info(f"Requirement {requirement} was not found. Trying to install...")
            info(check_output(f"pip install {requirement}", shell=True).decode())
