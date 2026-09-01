import unittest
from types import SimpleNamespace

import torch

from models.GRU_D import Model


class TestGRUDPreprocessing(unittest.TestCase):

    @staticmethod
    def _model():
        configs = SimpleNamespace(
            task_name="short_term_forecast",
            pred_len_max_irr=None,
            pred_len=2,
            seq_len_max_irr=None,
            seq_len=4,
            enc_in=2,
            d_model=8,
            n_classes=2,
            features="M",
        )
        return Model(configs)

    def test_model_forecast_shapes(self):
        model = self._model()
        x = torch.randn(3, 4, 2)

        result = model(x=x)

        self.assertEqual(result["pred"].shape, torch.Size((3, 2, 2)))
        self.assertEqual(result["true"].shape, torch.Size((3, 2, 2)))

    def test_empirical_mean_uses_observed_mask_convention(self):
        model = self._model()
        x = torch.tensor(
            [[[10.0, 100.0], [20.0, 200.0], [30.0, 300.0], [40.0, 400.0]]]
        )
        observed_mask = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]]
        )

        actual = model.calculate_empirical_mean(x, observed_mask)

        torch.testing.assert_close(actual, torch.tensor([[25.0, 200.0]]))

    def test_locf_is_causal_and_uses_mean_before_first_observation(self):
        model = self._model()
        x = torch.tensor(
            [[[10.0, 100.0], [20.0, 200.0], [30.0, 300.0], [40.0, 400.0]]]
        )
        observed_mask = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]]
        )
        empirical_mean = model.calculate_empirical_mean(x, observed_mask)

        actual = model.fill_locf(x, observed_mask, empirical_mean)

        expected = torch.tensor(
            [[[10.0, 200.0], [10.0, 200.0], [10.0, 200.0], [40.0, 200.0]]]
        )
        torch.testing.assert_close(actual, expected)

    def test_delta_accumulates_while_each_feature_is_missing(self):
        model = self._model()
        timestamps = torch.tensor([0.0, 0.1, 0.4, 0.8]).view(1, 4, 1)
        timestamps = timestamps.expand(-1, -1, 2)
        observed_mask = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]]
        )

        actual = model.convert_to_delta(timestamps, observed_mask)

        expected = torch.tensor(
            [[[0.0, 0.0], [0.1, 0.1], [0.4, 0.3], [0.8, 0.7]]]
        )
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
