# SineReLU

This implementation follows the piecewise definition in
[Wilder Rodrigues's original article](https://wilder-rodrigues.medium.com/sinerelu-an-alternative-to-the-relu-activation-function-e46a6199997d):

```text
SineReLU(x) = x                               if x > 0
              epsilon * (sin(x) - cos(x))      if x <= 0
```

The default `epsilon=0.0025` matches the CNN experiments in the article. Epsilon
is a fixed, finite, nonnegative hyperparameter; it is not learned. The activation
accepts floating-point tensors of any shape and operates independently on each
element. It has no running statistics and behaves identically in train/eval mode.
It does not modify its input in place, including when constructed through
`get_activation(..., inplace=True)`.

## Behavior at zero and gradients

Zero, including negative zero, selects the oscillating branch, so the result is
`-epsilon`. For `epsilon > 0`, the function is discontinuous at zero because its
right-hand limit is zero. Consequently it is not differentiable everywhere.
We preserve the article's piecewise definition rather than changing its boundary.

Autograd gives derivative `1` for positive inputs and
`epsilon * (cos(x) + sin(x))` for negative inputs. At exactly zero, PyTorch
follows the selected oscillating branch and returns gradient `epsilon`; this
is a branch-gradient convention, not a mathematical derivative at the jump.
Negative-branch derivatives can also be zero at isolated points.

For FP16/BF16 inputs, trigonometric operations and scaling use FP32; outputs
retain the input dtype and device. FP64 inputs retain double precision.

## Select SineReLU

Set `self.act = "sinerelu"` in an experiment before constructing the model, or
use the existing command-line configuration override:

```bash
python tools/train.py -n yolox-s -d 1 -b 4 --fp16 -expn yolox_s_sinerelu act sinerelu
```

This selects SineReLU for the backbone, neck, and hidden head convolutions.
The command assumes the usual YOLOX installation and dataset setup; batch size
must be checked on the training GPU.

To use a different amplitude directly:

```python
from yolox.models.network_blocks import SineReLU

activation = SineReLU(epsilon=0.025)
```

The `act sinerelu` factory selection uses the CNN default of `0.0025`.

## Verify

```bash
python -m unittest discover -s tests/models -p 'test_*.py' -v
```

The tests cover the piecewise values, zero boundary, analytical and numerical
gradients, low-precision inputs, and a YOLOX-S training/inference smoke check.
The CUDA autocast test runs when a CUDA GPU is available.
