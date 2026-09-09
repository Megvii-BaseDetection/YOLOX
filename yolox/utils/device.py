#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.
"""Accelerator selection that is not hard-wired to CUDA.

YOLOX assumed CUDA throughout: `torch.cuda.set_device`, `torch.cuda.amp`, and the
legacy `Tensor.type("torch.cuda.FloatTensor")` string API. None of those have a
meaning on Apple Silicon (MPS) or on a CPU-only host, so training there failed with
errors that pointed at tensor types rather than at the missing device.

These helpers centralise the choice so the rest of the codebase can ask for "the
device" instead of naming a vendor.
"""

import torch

__all__ = ["get_device_type", "get_device", "mps_available"]


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def get_device_type() -> str:
    """"cuda", "mps" or "cpu", preferring the fastest available."""
    if torch.cuda.is_available():
        return "cuda"
    if mps_available():
        return "mps"
    return "cpu"


def get_device(local_rank: int = 0) -> str:
    """Device string for this process. Only CUDA is rank-indexed; MPS and CPU are
    single-device, so appending a rank there would be invalid."""
    device_type = get_device_type()
    if device_type == "cuda":
        return "cuda:{}".format(local_rank)
    return device_type
