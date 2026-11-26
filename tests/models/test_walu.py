#!/usr/bin/env python3

import unittest

import torch
from torch import nn

from yolox.exp import get_exp
from yolox.models.network_blocks import BaseConv, WaLU, get_activation


class TestWaLU(unittest.TestCase):
    def test_activation_factory(self):
        self.assertIsInstance(get_activation("walu"), WaLU)
        self.assertIsInstance(get_activation("walu", inplace=False), WaLU)
        self.assertEqual(list(WaLU().parameters()), [])
        for name, activation in (
            ("silu", nn.SiLU), ("relu", nn.ReLU), ("lrelu", nn.LeakyReLU)
        ):
            with self.subTest(name=name):
                self.assertIsInstance(get_activation(name), activation)
        with self.assertRaises(AttributeError):
            get_activation("unknown")

    def test_conditional_means_across_channels_and_spatial_dimensions(self):
        x = torch.tensor([[[[-6.0, -2.0]], [[1.0, 9.0]]]])
        expected = torch.tensor([[[[-4.0, -2.0]], [[1.0, 5.0]]]])
        torch.testing.assert_close(WaLU()(x), expected)

    def test_missing_signs_and_zeros(self):
        for values, expected in (
            ([1.0, 3.0, 0.0], [1.0, 2.0, 0.0]),
            ([-1.0, -3.0, 0.0], [-1.0, -2.0, 0.0]),
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ):
            with self.subTest(values=values):
                x = torch.tensor(values, requires_grad=True).reshape(1, 1, 1, 3)
                y = WaLU()(x)
                torch.testing.assert_close(y.flatten(), torch.tensor(expected))
                gradient, = torch.autograd.grad(y.sum(), x)
                self.assertTrue(torch.isfinite(gradient).all())

    def test_independent_of_batch_composition(self):
        x = torch.tensor([[-6.0, -2.0, 1.0, 9.0], [-100.0, -20.0, 10.0, 90.0]])
        activation = WaLU()
        expected = torch.cat([activation(sample.unsqueeze(0)) for sample in x])
        torch.testing.assert_close(activation(x), expected)

    def test_gradients_flow_through_bounds(self):
        x = torch.tensor([[-6.0, -2.0, 1.0, 9.0]], requires_grad=True)
        WaLU()(x).sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([[0.5, 1.5, 1.5, 0.5]]))

    def test_numerical_gradients(self):
        # Avoid zero crossings and clipping boundaries, where derivatives
        # are not defined, when comparing to finite differences.
        x = torch.tensor([[-6.0, -2.0, 1.0, 9.0]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(WaLU(), (x,)))

    def test_large_low_precision_inputs(self):
        # Both sign counts and masked sums exceed the finite FP16 range.
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = torch.empty(1, 2, 256, 256, dtype=dtype)
                x[:, 0] = 2.0
                x[:, 1] = -4.0
                x.requires_grad_()
                y = WaLU()(x)
                self.assertEqual(y.dtype, dtype)
                self.assertEqual(y.device, x.device)
                torch.testing.assert_close(y, x)
                y.float().sum().backward()
                self.assertTrue(torch.isfinite(x.grad).all())

    def test_noncontiguous_input_is_not_modified(self):
        x = torch.tensor([[[[-6.0, -2.0], [1.0, 9.0]]]], dtype=torch.float64)
        x = x.transpose(2, 3)
        before = x.clone()
        y = get_activation("walu", inplace=True)(x)
        expected = torch.tensor([[[[-4.0, 1.0], [-2.0, 5.0]]]], dtype=x.dtype)
        torch.testing.assert_close(y, expected)
        torch.testing.assert_close(x, before)
        self.assertNotEqual(x.data_ptr(), y.data_ptr())

    def test_bounds_recomputed_in_evaluation(self):
        activation = WaLU().eval()
        x = torch.tensor([[-6.0, -2.0, 1.0, 9.0]])
        torch.testing.assert_close(activation(x), torch.tensor([[-4.0, -2.0, 1.0, 5.0]]))
        torch.testing.assert_close(activation(2 * x), 2 * activation(x))
        torch.testing.assert_close(activation.train()(x), activation.eval()(x))

    def test_requires_a_batched_input(self):
        with self.assertRaises(ValueError):
            WaLU()(torch.tensor([1.0, 2.0]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_mixed_precision(self):
        block = BaseConv(3, 8, 3, 1, act="walu").cuda()
        x = torch.randn(2, 3, 16, 16, device="cuda", requires_grad=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = block(x)
        self.assertEqual(y.dtype, torch.float16)
        self.assertTrue(torch.isfinite(y).all())
        y.float().square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(block.conv.weight.grad).all())

    def test_yolox_s_training_and_inference(self):
        exp = get_exp(exp_name="yolox-s")
        exp.merge(["act", "walu"])
        exp.num_classes = 3
        model = exp.get_model()
        blocks = [module for module in model.modules() if isinstance(module, BaseConv)]
        self.assertGreater(len(blocks), 0)
        self.assertTrue(all(isinstance(block.act, WaLU) for block in blocks))

        torch.manual_seed(0)
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
