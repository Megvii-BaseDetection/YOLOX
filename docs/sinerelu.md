# SineReLU

This is a direct PyTorch translation of the
[original keras-contrib source](https://github.com/keras-team/keras-contrib/blob/3fc5ef709e061416f4bc8a92ca3750c824b5d2b0/keras_contrib/layers/advanced_activations/sinerelu.py):

```python
wave = self.epsilon * (torch.sin(x) - torch.cos(x))
return torch.maximum(wave, x)
```

The default `epsilon=0.0025` matches the original source. Epsilon controls the wave
amplitude and is not learned. The activation operates independently on each
element, preserves the shape and dtype of floating-point inputs, and uses
PyTorch's native tensor operations and autograd. Train and eval behavior match.
Inputs are not modified in place.

## Behavior at zero and gradients

For the default positive epsilon, `SineReLU(0) = max(-epsilon, 0) = 0`, and the
gradient at zero is `1`. The identity branch also wins for some small negative
inputs, so selecting branches solely by the sign of the input is not equivalent.

Where the identity branch wins, the derivative is `1`; where the wave wins,
it is `epsilon * (cos(x) + sin(x))`. The function is continuous, but can have
kinks where the branches intersect. At ties, PyTorch applies its `maximum`
subgradient convention.

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

The tests cover the original maximum formula, zero behavior, analytical and numerical
gradients, low-precision inputs, and a YOLOX-S training/inference smoke check.
The CUDA autocast test runs when a CUDA GPU is available.
