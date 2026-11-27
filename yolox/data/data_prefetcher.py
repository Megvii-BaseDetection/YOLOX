#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader, device=None):
        self.loader = iter(loader)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, _, _ = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        if self.stream is None:
            self.next_input = self.next_input.to(self.device)
            self.next_target = self.next_target.to(self.device)
        else:
            with torch.cuda.stream(self.stream):
                self.next_input = self.next_input.to(self.device, non_blocking=True)
                self.next_target = self.next_target.to(self.device, non_blocking=True)

    def next(self):
        if self.stream is not None:
            current_stream = torch.cuda.current_stream(self.device)
            current_stream.wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if self.stream is not None:
            if input is not None:
                input.record_stream(current_stream)
            if target is not None:
                target.record_stream(current_stream)
        self.preload()
        return input, target
