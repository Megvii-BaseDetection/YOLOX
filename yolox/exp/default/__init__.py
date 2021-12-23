#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

# This file is used for package installation and find default exp file

import importlib
import sys
from pathlib import Path

_EXP_PATH = Path(__file__).resolve().parent.parent.parent.parent / "exps" / "default"

if _EXP_PATH.is_dir():
    # This is true only for in-place installation (pip install -e, setup.py develop),
    # where setup(package_dir=) does not work: https://github.com/pypa/setuptools/issues/230

    class _ExpFinder(importlib.abc.MetaPathFinder):
        
        def find_spec(self, name, path, target=None):
            if not name.startswith("yolox.exp.default"):
                return
            project_name = name.split(".")[-1] + ".py"
            target_file = _EXP_PATH / project_name
            if not target_file.is_file():
                return
            return importlib.util.spec_from_file_location(name, target_file)

    sys.meta_path.append(_ExpFinder())
