from glob import glob

import pytest
from PIL import Image
from pathlib import Path


@pytest.fixture(scope='session')
def test_images() -> list[Image.Image]:
    image_dir = Path(__file__).parent / 'data'
    image_files = sorted(glob(f'{image_dir}/*.jpg'))
    return [Image.open(file) for file in image_files]
