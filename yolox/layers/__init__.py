# Copyright (c) Megvii Inc. All rights reserved.

# import torch first to make jit op work without `ImportError of libc10.so`
import torch  # noqa

from .jit_ops import FastCocoEvalOp, JitOp

try:
    from .fast_coco_eval_api import CocoEvalOpt
except ImportError:  #  exception will be raised when users build yolox from source
    pass
