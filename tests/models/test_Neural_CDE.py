# Code from: https://github.com/Ladbaby/PyOmniTS
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from models.Neural_CDE import Model
from utils.configs import get_configs


class TestNeuralCDE(unittest.TestCase):

    @staticmethod
    def _configs(**overrides):
        values = {
            "task_name": "short_term_forecast",
            "pred_len_max_irr": None,
            "pred_len": 3,
            "enc_in": 1,
            "d_model": 12,
            "n_classes": 2,
            "features": "M",
            "neural_cde_adjoint": 0,
            "neural_cde_hidden_layers": 1,
            "neural_cde_hidden_width": 24,
            "neural_cde_solver": "rk4",
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    @staticmethod
    def _decay_data(n_samples=48, seed=1):
        generator = torch.Generator().manual_seed(seed)
        amplitude = 0.5 + torch.rand(
            n_samples, 1, 1, generator=generator
        )
        rate = 0.6 + torch.rand(n_samples, 1, 1, generator=generator)
        x_mark = torch.tensor([0.0, 0.12, 0.31, 0.5]).view(1, -1, 1)
        y_mark = torch.tensor([0.62, 0.79, 1.0]).view(1, -1, 1)
        x_mark = x_mark.repeat(n_samples, 1, 1)
        y_mark = y_mark.repeat(n_samples, 1, 1)
        x = amplitude * torch.exp(-rate * x_mark)
        y = amplitude * torch.exp(-rate * y_mark)
        return x, x_mark, torch.ones_like(x), y, y_mark, torch.ones_like(y)

    def test_control_uses_time_intensity_and_causal_locf(self):
        model = Model(self._configs(enc_in=2, d_model=8))
        values = torch.tensor(
            [[[1.0, 20.0], [1000.0, 21.0], [3.0, 1000.0]]]
        )
        mask = torch.tensor(
            [[[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]]
        )
        times = torch.tensor([[0.0, 0.2, 0.7]])

        control = model.backbone.build_control(values, times, mask)
        expected = torch.tensor(
            [
                [
                    [0.0, 1.0, 1.0, 1.0, 20.0],
                    [0.2, 1.0, 2.0, 1.0, 21.0],
                    [0.7, 2.0, 2.0, 3.0, 21.0],
                ]
            ]
        )
        torch.testing.assert_close(control, expected)

    def test_forward_is_finite_causal_and_differentiable(self):
        torch.manual_seed(3)
        model = Model(self._configs(enc_in=2, d_model=8))
        x = torch.randn(3, 4, 2)
        x_mark = torch.tensor([0.0, 0.2, 0.45, 0.7]).view(1, -1, 1)
        x_mark = x_mark.repeat(3, 1, 1)
        x_mask = torch.ones_like(x)
        x_mask[0, 1, 0] = 0
        y = torch.randn(3, 3, 2)
        y_mark = torch.tensor([0.8, 0.9, 1.0]).view(1, -1, 1)
        y_mark = y_mark.repeat(3, 1, 1)
        y_mask = torch.ones_like(y)

        output = model(x, x_mark, x_mask, y, y_mark, y_mask)
        changed_y_output = model(
            x, x_mark, x_mask, y + 1000.0, y_mark, y_mask
        )
        changed_masked_x = x.clone()
        changed_masked_x[0, 1, 0] = 1000.0
        changed_x_output = model(
            changed_masked_x, x_mark, x_mask, y, y_mark, y_mask
        )

        self.assertEqual(output["pred"].shape, y.shape)
        self.assertGreater(model.nfe, 0)
        self.assertTrue(torch.isfinite(output["pred"]).all())
        torch.testing.assert_close(output["pred"], changed_y_output["pred"])
        torch.testing.assert_close(output["pred"], changed_x_output["pred"])

        loss = torch.mean((output["pred"] - output["true"]).square())
        loss.backward()
        self.assertTrue(
            all(
                parameter.grad is None
                or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_learns_irregular_exponential_decay(self):
        torch.manual_seed(5)
        model = Model(self._configs())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        train_batch = self._decay_data(seed=1)
        test_batch = self._decay_data(seed=2)

        with torch.no_grad():
            initial_mse = torch.mean(
                (model(*test_batch)["pred"] - test_batch[3]).square()
            ).item()

        for _ in range(121):
            prediction = model(*train_batch)["pred"]
            loss = torch.mean((prediction - train_batch[3]).square())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            final_mse = torch.mean(
                (model(*test_batch)["pred"] - test_batch[3]).square()
            ).item()

        self.assertLess(final_mse, 0.01)
        self.assertLess(final_mse, 0.05 * initial_mse)

    def test_options_are_available_through_pyomnits_config(self):
        previous_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                os.chdir(temp_dir)
                configs = get_configs(
                    args=[
                        "--model_name",
                        "Neural_CDE",
                        "--model_id",
                        "Neural_CDE_test",
                        "--neural_cde_solver",
                        "dopri5",
                        "--neural_cde_hidden_layers",
                        "2",
                        "--neural_cde_hidden_width",
                        "64",
                        "--neural_cde_adjoint",
                        "1",
                    ]
                )
        finally:
            os.chdir(previous_cwd)

        self.assertEqual(configs.neural_cde_solver, "dopri5")
        self.assertEqual(configs.neural_cde_hidden_layers, 2)
        self.assertEqual(configs.neural_cde_hidden_width, 64)
        self.assertEqual(configs.neural_cde_adjoint, 1)


if __name__ == "__main__":
    unittest.main()
