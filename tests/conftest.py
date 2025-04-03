from glob import glob

import pytest
from PIL import Image
from pathlib import Path


@pytest.fixture(scope='session')
def test_image_files() -> list[str]:
    image_dir = Path(__file__).parent / 'data'
    return sorted(glob(f'{image_dir}/*.jpg'))
