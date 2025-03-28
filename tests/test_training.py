import hashlib
import subprocess
from pathlib import Path
from typing import Union

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
class TestTraining:
    """
    The idea of this test is to run a sample model training over just a few epochs in as deterministic a way as
    feasible, then validate the hash of the resulting checkpoint. Obviously the model will be junk, but the goal is to
    guard against regressions in the training logic or application of training parameters.

    I've managed to eliminate most nondeterminism from model training, but there is still some unusual
    "low-cardinality" nondeterminism; it seems there are exactly ten possible states for the resulting model weights.
    It would be great to find some way to ensure uniqueness of the output.
    """
    def test_training(self) -> None:
        subprocess.run(
            [
                'yolox',
                'train',
                '-c', 'yolox-s',  # Model config
                '-d', '1',  # Number of devices
                '-b', '8',  # Batch size
                '--fp16',  # Use mixed precision
                '-o',  # Occupy GPU memory first
                '-D', 'max_epoch=10',  # Train for 10 epochs
                '-D', 'no_aug_epochs=0',  # Avoid no-aug epochs
                '-D', 'seed=4171780',  # Set seed for reproducibility
                '-D', 'deterministic=True',  # Deterministic data loading
                '-D', 'data_num_workers=1',  # Number of data workers
            ],
            check=True,
        )
        sha = sha256sum('out/yolox_s/latest_ckpt.pth')
        assert sha in {
            "3e16caace3e2da177bcd0ce8b6787539476fcac68af7e7f773adf5e34cc3932f",
            "46287506ce10c7ceb97adfc6335d20e18d6743cf6787021268ca9324d46c7cb0",
            "945ca13b5b5e8a6be3b207dc5f8b83365cf070a7b95f511e648beb2670a59578",
            "b1ae463c821fb2dc9daa8c135c0fab9eac9df61edbaaa1a61a3a1373310db608",
            "b88fb45ab9662f414d3629384dcb6382b50caf853d3a6abee73f4f054ab653ca",
            "d312bcdf0341d626c76377f3b99628f5ad8886b6234f3df36a00d19324552fe7",
            "d8376aa22b168627956aa60ccc7d31b62a226707d0565d97f8d757b676903405",
            "d9ed40641b7b07706a9175134d211ddd2ae05d5188eb77955949811f8ca23fba",
            "f699433f20905f826c2c0681e485466affbf56c921e1c08260e97232f751d42c",
            "f83d47ba2611eab1370731b2993dfd7eace3d6f71aecdb69cc1cbfe3478477b2",
        }


def sha256sum(path: Union[Path, str]) -> str:
    """
    Compute the SHA256 hash of a file.
    """
    if isinstance(path, str):
        path = Path(path)

    h = hashlib.sha256()
    with open(path, 'rb') as file:
        while chunk := file.read(h.block_size):
            h.update(chunk)

    return h.hexdigest()
