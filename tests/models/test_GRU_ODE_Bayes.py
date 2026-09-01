# Code from: https://github.com/Ladbaby/PyOmniTS
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from models.GRU_ODE_Bayes import Model
from utils.configs import get_configs


class TestGRUODEBayes(unittest.TestCase):

    @staticmethod
    def _configs(**overrides):
        values = {
            "task_name": "short_term_forecast",
            "pred_len_max_irr": None,
            "pred_len": 3,
            "enc_in": 1,
            "d_model": 12,
            "dropout": 0.0,
            "n_classes": 2,
            "features": "M",
            "gru_ode_bayes_mixing": 1e-4,
            "gru_ode_bayes_p_hidden": 12,
            "gru_ode_bayes_prep_hidden": 6,
            "gru_ode_bayes_solver": "midpoint",
            "gru_ode_bayes_step_size": 0.05,
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
        self.assertEqual(output["pred_logvar"].shape, y.shape)
        self.assertGreater(model.nfe, 0)
        self.assertTrue(torch.isfinite(output["pred"]).all())
        torch.testing.assert_close(output["pred"], changed_y_output["pred"])
        torch.testing.assert_close(output["pred"], changed_x_output["pred"])

        output["loss"].backward()
        self.assertTrue(
            all(
                parameter.grad is None
                or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_gru_ode_vector_field_matches_reference_equations(self):
        torch.manual_seed(9)
        model = Model(self._configs(enc_in=2, d_model=7))
        cell = model.backbone.gru_ode
        prediction = torch.randn(5, 4)
        hidden = torch.randn(5, 7)

        actual = cell(prediction, hidden)
        update = torch.sigmoid(
            cell.lin_xz(prediction) + cell.lin_hz(hidden)
        )
        candidate = torch.tanh(
            cell.lin_xn(prediction) + cell.lin_hn(update * hidden)
        )
        expected = (1.0 - update) * (candidate - hidden)

        torch.testing.assert_close(actual, expected)

    def test_learns_irregular_exponential_decay(self):
        torch.manual_seed(5)
        model = Model(self._configs())
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        train_batch = self._decay_data(seed=1)
        test_batch = self._decay_data(seed=2)

        with torch.no_grad():
            initial_mse = torch.mean(
                (model(*test_batch)["pred"] - test_batch[3]).square()
            ).item()

        for _ in range(101):
            loss = model(*train_batch)["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

        with torch.no_grad():
            final_mse = torch.mean(
                (model(*test_batch)["pred"] - test_batch[3]).square()
            ).item()

        self.assertLess(final_mse, 0.005)
        self.assertLess(final_mse, 0.05 * initial_mse)

    def test_options_are_available_through_pyomnits_config(self):
        previous_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                os.chdir(temp_dir)
                configs = get_configs(
                    args=[
                        "--model_name",
                        "GRU_ODE_Bayes",
                        "--model_id",
                        "GRU_ODE_Bayes_test",
                        "--gru_ode_bayes_solver",
                        "midpoint",
                        "--gru_ode_bayes_step_size",
                        "0.025",
                        "--gru_ode_bayes_prep_hidden",
                        "25",
                    ]
                )
        finally:
            os.chdir(previous_cwd)

        self.assertEqual(configs.gru_ode_bayes_solver, "midpoint")
        self.assertEqual(configs.gru_ode_bayes_step_size, 0.025)
        self.assertEqual(configs.gru_ode_bayes_prep_hidden, 25)


if __name__ == "__main__":
    unittest.main()
