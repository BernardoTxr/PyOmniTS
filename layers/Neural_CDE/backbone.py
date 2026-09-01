# Code from: https://github.com/Ladbaby/PyOmniTS
# Code adapted from: https://github.com/patrick-kidger/NeuralCDE
# Copyright 2020 Patrick Kidger, James Morrill, James Foster, Terry Lyons
# Licensed under the Apache License, Version 2.0.

import torch
import torch.nn as nn
from torch import Tensor

try:
    import torchcde
except ImportError as error:  # pragma: no cover - exercised only without extras
    raise ImportError(
        "Neural_CDE requires torchcde. Install the PyOmniTS requirements "
        "or run `pip install torchcde==0.2.5`."
    ) from error


class CDEFunc(nn.Module):
    """Neural vector field with the final tanh used by the reference code."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        hidden_hidden_channels: int,
        num_hidden_layers: int,
    ) -> None:
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be at least 1")
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.linear_in = nn.Linear(hidden_channels, hidden_hidden_channels)
        self.hidden_linears = nn.ModuleList(
            nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
            for _ in range(num_hidden_layers - 1)
        )
        self.linear_out = nn.Linear(
            hidden_hidden_channels,
            hidden_channels * input_channels,
        )
        self.nfe = 0

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        self.nfe += 1
        z = torch.relu(self.linear_in(z))
        for layer in self.hidden_linears:
            z = torch.relu(layer(z))
        z = torch.tanh(self.linear_out(z))
        return z.view(
            *z.shape[:-1], self.hidden_channels, self.input_channels
        )


class NeuralCDEBackbone(nn.Module):
    """Neural CDE over time, observation intensity, and causal values."""

    def __init__(
        self,
        n_features: int,
        hidden_channels: int,
        hidden_hidden_channels: int,
        num_hidden_layers: int,
        solver: str,
        adjoint: bool,
    ) -> None:
        super().__init__()
        if solver not in {"euler", "rk4", "dopri5"}:
            raise ValueError("solver must be 'euler', 'rk4', or 'dopri5'")
        self.n_features = n_features
        self.input_channels = 1 + 2 * n_features
        self.hidden_channels = hidden_channels
        self.solver = solver
        self.adjoint = adjoint

        self.func = CDEFunc(
            input_channels=self.input_channels,
            hidden_channels=hidden_channels,
            hidden_hidden_channels=hidden_hidden_channels,
            num_hidden_layers=num_hidden_layers,
        )
        self.initial = nn.Linear(self.input_channels, hidden_channels)
        self.readout = nn.Linear(hidden_channels, n_features)

    @property
    def nfe(self) -> int:
        """Number of vector-field evaluations in the last solve."""
        return self.func.nfe

    @staticmethod
    def _locf(values: Tensor, mask: Tensor) -> Tensor:
        """Causal last-observation-carried-forward without masked leakage."""
        carried = torch.zeros_like(values[:, 0])
        filled = []
        for index in range(values.size(1)):
            observed = mask[:, index].bool()
            carried = torch.where(observed, values[:, index], carried)
            filled.append(carried)
        if not filled:
            return values.new_empty(values.shape)
        return torch.stack(filled, dim=1)

    def build_control(
        self,
        values: Tensor,
        times: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Build the reference time/intensity/value augmentation.

        Missing values are causally filled before interpolation. The
        cumulative intensity channels retain which variables were observed;
        padded rows therefore produce a constant control and no hidden drift.
        """
        if values.size(1) == 0:
            return values.new_empty(
                values.size(0), 0, self.input_channels
            )
        causal_values = self._locf(values, mask)
        intensity = mask.to(values.dtype).cumsum(dim=1)
        return torch.cat(
            [times.unsqueeze(-1), intensity, causal_values], dim=-1
        )

    def forward(self, control: Tensor) -> tuple[Tensor, Tensor]:
        if control.size(1) == 0:
            raise ValueError("Neural CDE needs at least one control point")

        coefficients = torchcde.linear_interpolation_coeffs(control)
        path = torchcde.LinearInterpolation(coefficients)
        z0 = self.initial(path.evaluate(path.interval[0]))
        self.func.nfe = 0

        solver_kwargs = {}
        if self.solver in {"euler", "rk4"}:
            solver_kwargs["options"] = {"step_size": 1.0}
        else:
            solver_kwargs.update(
                rtol=1e-3,
                atol=1e-5,
                options={"jump_t": path.grid_points},
            )
        hidden = torchcde.cdeint(
            X=path,
            func=self.func,
            z0=z0,
            t=path.grid_points,
            method=self.solver,
            adjoint=self.adjoint,
            **solver_kwargs,
        )
        return hidden, self.readout(hidden)
