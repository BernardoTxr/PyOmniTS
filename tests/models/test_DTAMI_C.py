import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from loss_fns.MSEWithRegularization import Loss
from models.DTAMI_C import Model
from utils.configs import get_configs


class TestDTAMIC(unittest.TestCase):

    @staticmethod
    def _configs(**overrides):
        values = {
            "task_name": "short_term_forecast",
            "pred_len_max_irr": None,
            "pred_len": 2,
            "enc_in": 3,
            "features": "M",
            "dtami_beta": 1.0,
            "dtami_hidden_units": 8,
            "dtami_n_traverse": 1,
            "dtami_h_ablation": "only_left",
            "dtami_initializer_type": "fixed-value",
            "dtami_init_r_min": 0.9,
            "dtami_init_r_max": 1.0,
            "dtami_init_f_min": 0.5,
            "dtami_init_f_max": 6.0,
            "dtami_init_r_value": 0.95,
            "dtami_init_theta_value": 1.0,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_forecast_forward_is_finite_and_regularized(self):
        model = Model(self._configs())
        x = torch.randn(2, 4, 3)
        x_mask = torch.tensor(
            [
                [[1, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 1]],
                [[1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 1]],
            ],
            dtype=x.dtype,
        )
        y = torch.randn(2, 2, 3)
        timestamps = torch.arange(6, dtype=x.dtype).view(1, 6, 1).repeat(2, 1, 1)

        outputs = model(
            x=x,
            x_mark=timestamps[:, :4],
            x_mask=x_mask,
            y=y,
            y_mark=timestamps[:, 4:],
            y_mask=torch.ones_like(y),
        )

        self.assertEqual(outputs["pred"].shape, y.shape)
        self.assertTrue(torch.isfinite(outputs["pred"]).all())
        self.assertEqual(outputs["regularization_loss"].ndim, 0)
        self.assertTrue(torch.isfinite(outputs["regularization_loss"]))

    def test_efficient_transition_matches_dense_source_formula(self):
        torch.manual_seed(7)
        model = Model(self._configs())
        h_t = torch.randn(5, model.hidden_units)
        deltat = torch.rand(5)

        actual = model.time_translation(h_t, deltat)

        k = model.hidden_units // 2
        scale = torch.exp(model.log_r).unsqueeze(0) ** deltat.unsqueeze(1)
        angle = model.theta.unsqueeze(0) * deltat.unsqueeze(1)
        dense_blocks = torch.zeros(5, model.hidden_units, model.hidden_units)
        even = torch.arange(0, model.hidden_units, 2)
        odd = even + 1
        dense_blocks[:, even, even] = scale * torch.cos(angle)
        dense_blocks[:, even, odd] = -scale * torch.sin(angle)
        dense_blocks[:, odd, even] = scale * torch.sin(angle)
        dense_blocks[:, odd, odd] = scale * torch.cos(angle)
        self.assertEqual(dense_blocks.shape[1], 2 * k)
        U = model.U.unsqueeze(0)
        W = U @ dense_blocks @ U.transpose(-1, -2)
        expected = torch.bmm(W, h_t.unsqueeze(-1)).squeeze(-1)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_regularized_mse_and_parseval_step(self):
        model = Model(self._configs())
        with torch.no_grad():
            model.U.mul_(1.01)
        before = model.parseval_loss().item()

        criterion = Loss(self._configs())
        result = criterion(
            pred=torch.zeros(1, 1, 1),
            true=torch.ones(1, 1, 1),
            regularization_loss=model.parseval_loss(),
        )
        self.assertAlmostEqual(result["loss"].item(), 1.0 + before, places=5)

        U_before = model.U.detach().clone()
        expected = (
            (1.0 + model.beta) * U_before
            - model.beta * (
                (U_before @ U_before.transpose(0, 1)) @ U_before
            )
        )
        model.post_step()
        torch.testing.assert_close(model.U, expected)
        self.assertIsNotNone(model.orthogonality_error)
        self.assertTrue(torch.isfinite(model.parseval_loss()))

    def test_half_beta_parseval_step_contracts_small_orthogonality_error(self):
        model = Model(self._configs(dtami_beta=0.5))
        with torch.no_grad():
            model.U.mul_(1.01)
        identity = torch.eye(model.hidden_units)
        before = torch.linalg.matrix_norm(model.U.T @ model.U - identity)

        model.post_step()

        after = torch.linalg.matrix_norm(model.U.T @ model.U - identity)
        self.assertLess(after.item(), before.item())

    def test_dtami_c_options_are_available_through_pyomnits_config(self):
        previous_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                os.chdir(temp_dir)
                configs = get_configs(
                    args=[
                        "--model_name", "DTAMI_C",
                        "--model_id", "DTAMI_C_test",
                        "--dtami_beta", "0.5",
                        "--dtami_hidden_units", "128",
                        "--dtami_h_ablation", "only_left",
                    ]
                )
        finally:
            os.chdir(previous_cwd)

        self.assertEqual(configs.dtami_beta, 0.5)
        self.assertEqual(configs.dtami_hidden_units, 128)
        self.assertEqual(configs.dtami_h_ablation, "only_left")


if __name__ == "__main__":
    unittest.main()
