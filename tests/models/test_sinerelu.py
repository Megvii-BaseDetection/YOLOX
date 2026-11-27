#!/usr/bin/env python3

import math
import unittest

import torch

from yolox.exp import get_exp
from yolox.models.network_blocks import BaseConv, SineReLU, get_activation


class TestSineReLU(unittest.TestCase):
    def test_activation_factory(self):
        for inplace in (True, False):
            with self.subTest(inplace=inplace):
                activation = get_activation("sinerelu", inplace=inplace)
                self.assertIsInstance(activation, SineReLU)
                self.assertEqual(activation.epsilon, 0.0025)
                self.assertEqual(list(activation.parameters()), [])
                self.assertEqual(list(activation.buffers()), [])

    def test_matches_keras_maximum(self):
        values = [-7 * math.pi / 4, -math.pi, -math.pi / 4, -0.1, 0.0, 1e-8, 2.0]
        x = torch.tensor(values, dtype=torch.float64)
        for epsilon in (0.0025, 0.025, 0.1):
            with self.subTest(epsilon=epsilon):
                expected = torch.tensor([
                    max(epsilon * (math.sin(value) - math.cos(value)), value)
                    for value in values
                ], dtype=x.dtype)
                torch.testing.assert_close(SineReLU(epsilon)(x), expected)

    def test_identity_branch_at_and_near_zero(self):
        x = torch.tensor([-0.002, -1e-12, -0.0, 0.0, 1e-12], dtype=torch.float64)
        torch.testing.assert_close(SineReLU()(x), x)

    def test_analytical_gradients(self):
        epsilon = 0.0025
        values = [-math.pi, -0.4, -1e-8, 0.0, 1e-8, 2.0]
        x = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        SineReLU(epsilon)(x).sum().backward()
        # The identity branch also wins for sufficiently small negative inputs.
        expected = torch.tensor([
            1.0 if value > epsilon * (math.sin(value) - math.cos(value))
            else epsilon * (math.cos(value) + math.sin(value))
            for value in values
        ], dtype=x.dtype)
        torch.testing.assert_close(x.grad, expected)

    def test_numerical_gradients(self):
        x = torch.tensor(
            [-6.0, -2.0, -0.25, 0.0, 0.25, 9.0], dtype=torch.float64, requires_grad=True
        )
        self.assertTrue(torch.autograd.gradcheck(SineReLU(), (x,)))

    def test_zero_epsilon_reduces_to_relu(self):
        x = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
        torch.testing.assert_close(SineReLU(epsilon=0)(x), x.relu())

    def test_shape_dtype_and_device(self):
        generator = torch.Generator().manual_seed(7)
        for shape in ((), (0,), (2, 0, 3), (5,), (2, 3), (2, 3, 4, 5)):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(shape=shape, dtype=dtype):
                    x = torch.randn(shape, generator=generator, dtype=dtype)
                    y = SineReLU()(x)
                    self.assertEqual(y.shape, x.shape)
                    self.assertEqual(y.dtype, x.dtype)
                    self.assertEqual(y.device, x.device)
                    self.assertTrue(torch.isfinite(y).all())

    def test_low_precision_matches_reference(self):
        epsilon = 0.0025
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = torch.tensor(
                    [-60000.0, -6.0, -2.0, -0.25, 0.0, 0.25, 2.0, 60000.0],
                    dtype=dtype, requires_grad=True,
                )
                reference = torch.tensor([
                    max(epsilon * (math.sin(value) - math.cos(value)), value)
                    for value in x.detach().double().tolist()
                ], dtype=dtype)
                y = SineReLU(epsilon)(x)
                self.assertEqual(y.dtype, dtype)
                torch.testing.assert_close(y, reference)
                y.float().sum().backward()
                self.assertTrue(torch.isfinite(x.grad).all())

    def test_noncontiguous_input_is_not_modified(self):
        x = torch.tensor([[[-6.0, -2.0], [1.0, 9.0]]]).transpose(1, 2)
        before = x.clone()
        y = get_activation("sinerelu", inplace=True)(x)
        torch.testing.assert_close(x, before)
        torch.testing.assert_close(y, SineReLU()(x.contiguous()))
        self.assertNotEqual(x.data_ptr(), y.data_ptr())

    def test_train_and_eval_match(self):
        activation = SineReLU()
        x = torch.tensor([-6.0, -2.0, 0.0, 1.0, 9.0])
        torch.testing.assert_close(activation.train()(x), activation.eval()(x))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_mixed_precision(self):
        block = BaseConv(3, 8, 3, 1, act="sinerelu").cuda()
        x = torch.randn(2, 3, 16, 16, device="cuda", requires_grad=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = block(x)
        self.assertEqual(y.dtype, torch.float16)
        self.assertTrue(torch.isfinite(y).all())
        y.float().square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(block.conv.weight.grad).all())

    def test_yolox_s_training_and_inference(self):
        torch.manual_seed(0)
        exp = get_exp(exp_name="yolox-s")
        exp.merge(["act", "sinerelu"])
        exp.num_classes = 3
        model = exp.get_model()
        blocks = [module for module in model.modules() if isinstance(module, BaseConv)]
        self.assertGreater(len(blocks), 0)
        self.assertTrue(all(isinstance(block.act, SineReLU) for block in blocks))

        x = torch.randn(2, 3, 64, 64, requires_grad=True)
        targets = torch.tensor([[[0.0, 32.0, 32.0, 16.0, 16.0]],
                                [[1.0, 20.0, 20.0, 12.0, 12.0]]])
        losses = model(x, targets)
        self.assertTrue(torch.isfinite(losses["total_loss"]))
        losses["total_loss"].backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

        model.eval()
        with torch.no_grad():
            predictions = model(x.detach())
        self.assertEqual(predictions.shape, (2, 84, 8))
        self.assertTrue(torch.isfinite(predictions).all())


if __name__ == "__main__":
    unittest.main()
