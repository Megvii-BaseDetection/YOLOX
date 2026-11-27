from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset

from yolox.core import Trainer
from yolox.data import DataPrefetcher
from yolox.evaluators import COCOEvaluator
from yolox.exp import get_exp
from yolox.utils import configure_device, get_device


def arguments(**overrides):
    values = dict(device="auto", num_machines=1, devices=None, fp16=False, occupy=False)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_auto_cpu_disables_cuda_options():
    args = arguments(fp16=True, occupy=True)
    with patch("torch.cuda.is_available", return_value=False):
        assert configure_device(args) == 1
    assert args.device == "cpu"
    assert not args.fp16
    assert not args.occupy


def test_requested_cuda_without_gpu_has_clear_error():
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(ValueError, match="CUDA is unavailable"):
            get_device("cuda")


def test_cuda_configuration_keeps_amp():
    args = arguments(device="cuda", devices=1, fp16=True)
    with patch("torch.cuda.is_available", return_value=True), \
            patch("torch.cuda.device_count", return_value=1):
        assert configure_device(args) == 1
    assert args.device == "cuda"
    assert args.fp16


def test_cpu_prefetch_does_not_create_cuda_stream():
    images = torch.randn(2, 3, 64, 64)
    targets = torch.zeros(2, 1, 5)
    with patch("torch.cuda.Stream", side_effect=AssertionError("unexpected CUDA stream")):
        prefetcher = DataPrefetcher([(images, targets, None, None)], device="cpu")
        result_images, result_targets = prefetcher.next()
        torch.testing.assert_close(result_images, images)
        torch.testing.assert_close(result_targets, targets)
        assert prefetcher.next() == (None, None)


def test_random_resize_without_cuda():
    exp = get_exp(exp_name="yolox-s")
    exp.input_size = (64, 64)
    exp.multiscale_range = 0
    with patch("torch.cuda.current_device", side_effect=AssertionError("unexpected CUDA")):
        assert exp.random_resize(None, 0, 0, False) == (64, 64)


class TinyDataset(Dataset):
    """Two annotated synthetic images for the complete CPU train/eval path."""

    def __init__(self):
        self.class_ids = [1, 2, 3]
        self.coco = COCO()
        self.coco.dataset = {
            "info": {},
            "images": [{"id": i, "height": 64, "width": 64} for i in range(2)],
            "categories": [{"id": i + 1, "name": str(i)} for i in range(3)],
            "annotations": [
                {"id": i + 1, "image_id": i, "category_id": 1,
                 "bbox": [24, 24, 16, 16], "area": 256, "iscrowd": 0}
                for i in range(2)
            ],
        }
        self.coco.createIndex()

    def __len__(self):
        return 2

    def __getitem__(self, index):
        image = torch.ones(3, 64, 64) * (index + 1)
        target = torch.tensor([[0.0, 32.0, 32.0, 16.0, 16.0]])
        return image, target, (64, 64), index


class TinyLoader(DataLoader):
    def close_mosaic(self):
        pass


def test_cpu_training_evaluation_and_checkpoint(tmp_path):
    exp = get_exp(exp_name="yolox-s")
    exp.num_classes = 3
    exp.act = "walu"
    exp.input_size = exp.test_size = (64, 64)
    exp.max_epoch = 1
    exp.warmup_epochs = 0
    exp.no_aug_epochs = 0
    exp.output_dir = str(tmp_path)
    exp.test_conf = 0.0
    loader = TinyLoader(TinyDataset(), batch_size=2)
    exp.get_data_loader = lambda **kwargs: loader
    exp.get_evaluator = lambda **kwargs: COCOEvaluator(loader, (64, 64), 0.0, 0.65, 3)
    args = arguments(
        device="cpu", batch_size=2, experiment_name="cpu-smoke", logger="tensorboard",
        cache=None, resume=False, ckpt=None, exp_file=None, start_epoch=None,
    )
    trainer = Trainer(exp, args)
    try:
        trainer.train()
        for name in ("latest", "last_epoch"):
            checkpoint = tmp_path / "cpu-smoke" / f"{name}_ckpt.pth"
            assert checkpoint.is_file()
            state = torch.load(checkpoint, map_location="cpu")
            assert "model" in state
            assert all(torch.isfinite(value).all() for value in state["model"].values())
    finally:
        if hasattr(trainer, "tblogger"):
            trainer.tblogger.close()
