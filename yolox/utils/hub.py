from logging import info, basicConfig, INFO
from subprocess import check_output
from typing import List

import pkg_resources as pkg


def ensure_installed(requirements: List[str]) -> None:
    """
        Checks for required packages and installs them if possible

        Args:
            requirements: list of required python packages

        Returns:
            None
    """
    for requirement in requirements:
        try:
            pkg.require(requirement)
        except Exception:
            basicConfig(format="%(message)s", level=INFO)
            info(f"Requirement {requirement} was not found. Trying to install...")
            info(check_output(f"pip install {requirement}", shell=True).decode())
