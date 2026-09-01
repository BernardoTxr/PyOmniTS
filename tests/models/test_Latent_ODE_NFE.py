# Code from: https://github.com/Ladbaby/PyOmniTS
import unittest

import torch

from layers.Latent_ODE.ode_func import ODEFunc


class TestLatentODENFE(unittest.TestCase):
    def test_vector_field_counts_function_evaluations(self):
        function = ODEFunc(
            input_dim=1,
            latent_dim=3,
            ode_func_net=torch.nn.Linear(3, 3),
            device=torch.device("cpu"),
        )
        state = torch.ones(2, 3)

        function(torch.tensor(0.0), state)
        function(torch.tensor(0.1), state)

        self.assertEqual(function.nfe, 2)


if __name__ == "__main__":
    unittest.main()
