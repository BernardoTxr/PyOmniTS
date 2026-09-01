# Code from: https://github.com/Ladbaby/PyOmniTS
# Code adapted from: https://github.com/edebrouwer/gru_ode_bayes
# Copyright (c) 2022 Edward De Brouwer
# Licensed under the MIT License.

import math

import torch
import torch.nn as nn
from torch import Tensor


class GRUODECell(nn.Module):
    """Negative-feedback GRU-ODE cell from De Brouwer et al. (2019)."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.lin_xz = nn.Linear(input_size, hidden_size)
        self.lin_xn = nn.Linear(input_size, hidden_size)
        self.lin_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.nfe = 0

    def forward(self, p: Tensor, h: Tensor) -> Tensor:
        self.nfe += 1
        z = torch.sigmoid(self.lin_xz(p) + self.lin_hz(h))
        n = torch.tanh(self.lin_xn(p) + self.lin_hn(z * h))
        return (1.0 - z) * (n - h)


class GRUObservationCellLogvar(nn.Module):
    """Bayesian jump that incorporates a partially observed vector."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        prep_hidden: int,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.prep_hidden = prep_hidden
        self.gru = nn.GRUCell(prep_hidden * input_size, hidden_size)

        std = math.sqrt(2.0 / (4 + prep_hidden))
        self.w_prep = nn.Parameter(
            std * torch.randn(input_size, 4, prep_hidden)
        )
        self.bias_prep = nn.Parameter(
            torch.full((input_size, prep_hidden), 0.1)
        )

    @staticmethod
    def gaussian_nll(
        prediction: Tensor,
        values: Tensor,
        mask: Tensor,
    ) -> Tensor:
        mean, logvar = prediction.chunk(2, dim=-1)
        logvar = logvar.clamp(min=-10.0, max=10.0)
        error = (values - mean) * torch.exp(-0.5 * logvar)
        return 0.5 * (
            error.square() + logvar + math.log(2.0 * math.pi)
        ) * mask

    def forward(
        self,
        h: Tensor,
        prediction: Tensor,
        values: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        mean, logvar = prediction.chunk(2, dim=-1)
        logvar = logvar.clamp(min=-10.0, max=10.0)
        error = (values - mean) * torch.exp(-0.5 * logvar)

        prepared = torch.stack([values, mean, logvar, error], dim=-1)
        prepared = torch.einsum(
            "bdc,dcp->bdp", prepared, self.w_prep
        )
        prepared = torch.relu(prepared + self.bias_prep)
        prepared = (prepared * mask.unsqueeze(-1)).flatten(start_dim=1)

        updated = self.gru(prepared, h)
        has_observation = mask.bool().any(dim=-1, keepdim=True)
        h = torch.where(has_observation, updated, h)
        return h, self.gaussian_nll(prediction, values, mask)


class GRUODEBayesBackbone(nn.Module):
    """Dense-batch adaptation of the official GRU-ODE-Bayes model.

    PyOmniTS batches observations on a padded time axis. This class applies
    the same alternating continuous GRU-ODE propagation and discrete
    GRU-Bayes update as the event-table reference implementation, while
    allowing every sample and channel to have its own observation mask.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p_hidden: int,
        prep_hidden: int,
        mixing: float,
        solver: str,
        step_size: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if solver not in {"euler", "midpoint"}:
            raise ValueError("solver must be 'euler' or 'midpoint'")
        if step_size <= 0:
            raise ValueError("step_size must be positive")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mixing = mixing
        self.solver = solver
        self.step_size = step_size

        self.p_model = nn.Sequential(
            nn.Linear(hidden_size, p_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(p_hidden, 2 * input_size),
        )
        self.gru_ode = GRUODECell(2 * input_size, hidden_size)
        self.gru_obs = GRUObservationCellLogvar(
            input_size, hidden_size, prep_hidden
        )
        self.initial_hidden = nn.Parameter(torch.zeros(hidden_size))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.05)

    @property
    def nfe(self) -> int:
        """Number of GRU-ODE vector-field evaluations in the last pass."""
        return self.gru_ode.nfe

    def reset_nfe(self) -> None:
        self.gru_ode.nfe = 0

    def _propagate(
        self,
        h: Tensor,
        prediction: Tensor,
        delta_t: Tensor,
    ) -> tuple[Tensor, Tensor]:
        max_delta = float(delta_t.detach().max())
        n_steps = max(1, math.ceil(max_delta / self.step_size))
        step = (delta_t / n_steps).unsqueeze(-1)

        for _ in range(n_steps):
            if self.solver == "euler":
                h = h + step * self.gru_ode(prediction, h)
            else:
                midpoint_h = h + 0.5 * step * self.gru_ode(prediction, h)
                midpoint_prediction = self.p_model(midpoint_h)
                h = h + step * self.gru_ode(
                    midpoint_prediction, midpoint_h
                )
            prediction = self.p_model(h)
        return h, prediction

    @staticmethod
    def _post_update_kl(
        prediction: Tensor,
        values: Tensor,
        mask: Tensor,
        observation_std: float = 1e-2,
    ) -> Tensor:
        mean, logvar = prediction.chunk(2, dim=-1)
        logvar = logvar.clamp(min=-10.0, max=10.0)
        variance = torch.exp(logvar)
        observation_variance = observation_std**2
        kl = (
            math.log(observation_std)
            - 0.5 * logvar
            + (variance + (mean - values).square())
            / (2.0 * observation_variance)
            - 0.5
        )
        return kl * mask

    def forward(
        self,
        context_values: Tensor,
        context_times: Tensor,
        context_mask: Tensor,
        target_times: Tensor,
        target_values: Tensor | None = None,
        target_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Filter context observations and forecast without target leakage.

        Returns the target mean, target log-variance, and the normalized
        official pre-jump plus post-jump training objective. Target values are
        used only to score the predictive distribution; they never update the
        hidden state.
        """
        batch_size = context_values.size(0)
        h = self.initial_hidden.unsqueeze(0).expand(batch_size, -1)
        prediction = self.p_model(h)
        self.reset_nfe()

        total_nll = context_values.new_zeros(())
        total_kl = context_values.new_zeros(())
        n_nll = context_values.new_zeros(())
        n_kl = context_values.new_zeros(())
        previous_time = context_times.new_zeros(batch_size)

        for index in range(context_values.size(1)):
            current_time = context_times[:, index]
            delta_t = (current_time - previous_time).clamp_min(0.0)
            h, prediction = self._propagate(h, prediction, delta_t)

            values = context_values[:, index]
            mask = context_mask[:, index].to(values.dtype)
            h, nll = self.gru_obs(h, prediction, values, mask)
            total_nll = total_nll + nll.sum()
            n_nll = n_nll + mask.sum()

            prediction = self.p_model(h)
            total_kl = total_kl + self._post_update_kl(
                prediction, values, mask
            ).sum()
            n_kl = n_kl + mask.sum()
            previous_time = torch.maximum(previous_time, current_time)

        target_predictions = []
        for index in range(target_times.size(1)):
            current_time = target_times[:, index]
            delta_t = (current_time - previous_time).clamp_min(0.0)
            h, prediction = self._propagate(h, prediction, delta_t)
            target_predictions.append(prediction)

            if target_values is not None and target_mask is not None:
                mask = target_mask[:, index].to(context_values.dtype)
                nll = self.gru_obs.gaussian_nll(
                    prediction, target_values[:, index], mask
                )
                total_nll = total_nll + nll.sum()
                n_nll = n_nll + mask.sum()
            previous_time = torch.maximum(previous_time, current_time)

        if target_predictions:
            target_prediction = torch.stack(target_predictions, dim=1)
        else:
            target_prediction = prediction.new_empty(
                batch_size, 0, 2 * self.input_size
            )
        target_mean, target_logvar = target_prediction.chunk(2, dim=-1)
        target_logvar = target_logvar.clamp(min=-10.0, max=10.0)

        loss = total_nll / n_nll.clamp_min(1.0)
        loss = loss + self.mixing * total_kl / n_kl.clamp_min(1.0)
        return target_mean, target_logvar, loss
