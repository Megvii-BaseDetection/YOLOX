# Poetry environment

Use Python 3.13.7 or a newer 3.13/3.14 release, and Poetry 2.3 or newer.
The supported Python range is `>=3.13.7,<3.15`; later minors need a new wheel
compatibility check. PyTorch and torchvision are pinned to the compatible pair
`2.10.0` / `0.25.0`, with one committed lock for macOS, Linux CPU, and Linux CUDA.
The wheel combinations follow the [official PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/).

## Install

```bash
poetry env use 3.13.7
poetry sync
```

This creates an editable project environment. It does not replace an existing
`venv` directory; configure your IDE to use the path from `poetry env info --path`.
If using pyenv, install Python 3.13.7 first; `.python-version` selects that version.
Use `poetry run` before Python commands to select the project environment.

| Machine | Install | Default execution |
| --- | --- | --- |
| MacBook with Apple silicon | `poetry sync` | CPU |
| Linux without NVIDIA CUDA | `poetry sync` | CPU |
| Linux with an NVIDIA GPU | `poetry sync --extras cuda` | CUDA when available |

The default Linux install uses the official CPU wheels. The CUDA extra uses the
official CUDA 12.6 wheels, including the runtime libraries; an NVIDIA driver
compatible with CUDA 12.6 is required. Installing a system CUDA toolkit or NVCC
is unnecessary for normal training. This CUDA build is chosen for the RTX 2080
rather than a CUDA 13 build. CUDA wheels here target Linux x86-64.

The macOS wheels target Apple silicon. Intel Macs are not supported by this
Python/PyTorch combination. macOS execution currently uses CPU, not Metal/MPS.
Linux uses headless OpenCV for server installs; use `--save_result` for demos.

Switch between CPU and CUDA installs with `poetry sync` and
`poetry sync --extras cuda`. Poetry cannot detect GPU hardware while resolving
dependencies; the extra chooses the wheels, and runtime detection chooses the
execution device. `poetry.toml` keeps installation from re-resolving these
mutually exclusive dependencies.

## Run and verify

```bash
poetry run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
poetry run pytest -q
poetry run python -m yolox.tools.train --help
poetry run python -m yolox.tools.eval --help
```

With your dataset configured, examples are:

```bash
poetry run python tools/train.py -n yolox-s --device cpu -b 2 act walu
poetry run python tools/train.py -n yolox-s --device cuda -d 1 -b 4 --fp16 act walu
```

Training and evaluation default to `--device auto`. You can explicitly choose
`cpu` or `cuda`. CPU runs use one process and FP32, even if `--fp16` was supplied.
CUDA runs retain multi-GPU training, asynchronous prefetching, and mixed precision.
Batch size still needs to be measured on the training machine.

Standard COCO evaluation uses pycocotools on every platform. The optional C++
evaluation code remains in the package, but normal installation and evaluation
do not compile or require it.
Model summaries use PyTorch's built-in FLOP counter, so no legacy profiler
dependency is needed.

## Optional tools

```bash
poetry sync --with export
poetry sync --with docs
poetry run sphinx-build -b html docs /tmp/yolox-docs
poetry build
```

Include `--extras cuda` with these commands on your CUDA training machine.
ONNX tools and Sphinx are optional dependency groups. TensorRT/torch2trt and
MegEngine deployment integrations require their own vendor-specific environments.
W&B and MLflow remain optional integrations (`poetry add wandb` / `poetry add mlflow`).

Update dependencies with `poetry update`, and commit `pyproject.toml` and
`poetry.lock` together. CI installs the lock and runs tests/builds on Linux and
Apple-silicon macOS with Python 3.13.7 and 3.14.
