# WaLU: Winsorized Adaptive Linear Unit

WaLU clips activations to their mean negative and mean positive values. For
each sample in an `(N, C, H, W)` tensor, it computes one pair of bounds across
`(C, H, W)`. Other batched shapes `(N, ...)` are supported in the same way.
Zeros do not contribute to either mean. If a sign is absent, its bound is zero.

For example, `[-6, -2, 0, 1, 9]` has bounds `-4` and `5`, giving
`[-4, -2, 0, 1, 5]`.

The bounds are recomputed on every forward pass, including evaluation, and
gradients flow through the means. WaLU has no learned parameters or running
statistics and does not modify its input in place. FP16/BF16 inputs use FP32
statistics and division to avoid overflow; the output retains the input dtype.

## Select WaLU

Set `self.act = "walu"` in an experiment before constructing the model, or use
YOLOX's existing command-line configuration override:

```bash
python tools/train.py -n yolox-s -d 1 -b 4 --fp16 -expn yolox_s_walu act walu
```

This enables WaLU in the CSPDarknet backbone, PAN neck, and hidden detection-head
convolutions. Prediction decoding and output constraints are unchanged. Existing
experiments still default to SiLU. The command assumes the usual YOLOX installation
and dataset setup; batch size must be checked on the training GPU.

## Verify

From the repository root with YOLOX's runtime dependencies installed:

```bash
python -m unittest discover -s tests/models -p test_walu.py -v
```

The tests cover sign handling, per-sample reductions, gradients through bounds,
large mixed-precision inputs, and YOLOX-S training/inference smoke checks. The
CUDA autocast test runs when a CUDA GPU is available.
