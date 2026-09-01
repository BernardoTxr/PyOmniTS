import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from models.DTAMI_CIRC import Model
from utils.configs import get_configs


class TestDTAMICIRC(unittest.TestCase):

    @staticmethod
    def _configs(**overrides):
        values = {
            "task_name": "short_term_forecast",
            "pred_len_max_irr": None,
            "pred_len": 2,
            "enc_in": 2,
            "features": "M",
            "dtami_hidden_units": 8,
            "dtami_n_traverse": 1,
            "dtami_h_ablation": "only_left",
            "dtami_initializer_type": "fixed-value",
            "dtami_init_r_min": 0.9,
            "dtami_init_r_max": 1.0,
            "dtami_init_f_min": 0.5,
            "dtami_init_f_max": 6.0,
            "dtami_init_r_value": 1.0,
            "dtami_init_theta_value": 1.0,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_backbone_settings_are_configurable_and_only_left_is_used(self):
        model = Model(self._configs(dtami_hidden_units=16, dtami_n_traverse=2))

        self.assertEqual(model.hidden_units, 16)
        self.assertEqual(model.n_traverse, 2)
        self.assertEqual(model.h_ablation, "only_left")
        self.assertEqual(model.init_eigenvalues_config["type"], "fixed-value")

        features = torch.randn(1, 4, 4)
        deltat = torch.ones(1, 3)
        h_0 = torch.zeros(1, 4, model.hidden_units)
        context_mask = torch.tensor([[True, True, False, False]])
        h_out, _, h_left, _ = model.process_sequence(
            features, deltat, h_0, context_mask, return_components=True
        )
        torch.testing.assert_close(h_out, h_left)

        h_fast = model.process_sequence(
            features, deltat, h_0, context_mask, return_components=False
        )
        torch.testing.assert_close(h_fast, h_left)

    def test_forecast_context_is_derived_from_observation_mask(self):
        model = Model(self._configs())
        captured = {}

        def capture_context(data, timestamps, h_0, context_mask, return_components=False):
            captured["context_mask"] = context_mask.clone()
            return torch.zeros(
                data.size(0), data.size(1), model.input_size,
                dtype=data.dtype, device=data.device,
            )

        model._dtamicirc_forward = capture_context

        x = torch.randn(2, 3, 2)
        x_mask = torch.tensor(
            [
                [[1, 0], [0, 0], [0, 1]],
                [[0, 0], [0, 0], [1, 1]],
            ],
            dtype=x.dtype,
        )
        y = torch.randn(2, 2, 2)
        x_mark = torch.tensor([[[0.0], [0.2], [0.4]]] * 2)
        y_mark = torch.tensor([[[0.6], [0.8]]] * 2)

        result = model(
            x=x,
            x_mark=x_mark,
            x_mask=x_mask,
            y=y,
            y_mark=y_mark,
            y_mask=torch.ones_like(y),
        )

        expected = torch.tensor(
            [
                [True, False, True, False, False],
                [False, False, True, False, False],
            ]
        )
        torch.testing.assert_close(captured["context_mask"], expected)
        self.assertEqual(result["pred"].shape, (2, 2, 2))

    def test_dtami_options_are_available_through_pyomnits_config(self):
        previous_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                os.chdir(temp_dir)
                configs = get_configs(
                    args=[
                        "--model_name", "DTAMI_CIRC",
                        "--model_id", "DTAMI_CIRC_test",
                        "--dtami_hidden_units", "128",
                        "--dtami_n_traverse", "3",
                        "--dtami_h_ablation", "only_left",
                        "--dtami_initializer_type", "frequency-log-uniform",
                        "--dtami_init_f_min", "0.25",
                        "--dtami_init_f_max", "8.0",
                    ]
                )
        finally:
            os.chdir(previous_cwd)

        self.assertEqual(configs.dtami_hidden_units, 128)
        self.assertEqual(configs.dtami_n_traverse, 3)
        self.assertEqual(configs.dtami_h_ablation, "only_left")
        self.assertEqual(configs.dtami_initializer_type, "frequency-log-uniform")
        self.assertEqual(configs.dtami_init_f_min, 0.25)
        self.assertEqual(configs.dtami_init_f_max, 8.0)


if __name__ == "__main__":
    unittest.main()
