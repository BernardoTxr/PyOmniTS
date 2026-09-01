# Code from: https://github.com/Ladbaby/PyOmniTS
# Code adapted from: https://github.com/edebrouwer/gru_ode_bayes
# Copyright (c) 2022 Edward De Brouwer
# Licensed under the MIT License.

import torch
import torch.nn as nn
from einops import repeat
from torch import Tensor

from layers.GRU_ODE_Bayes import GRUODEBayesBackbone
from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


class Model(nn.Module):
    """
    - paper: "GRU-ODE-Bayes: Continuous Modeling of Sporadically-Observed
      Time Series" (NeurIPS 2019)
    - paper link: https://arxiv.org/abs/1905.12374
    - code adapted from: https://github.com/edebrouwer/gru_ode_bayes

    The adaptor preserves the reference model's negative-feedback GRU-ODE,
    Gaussian observation model, pre-observation likelihood, and GRU-Bayes
    jump. Forecast targets are scored but never fed into the hidden state.
    """

    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.configs = configs
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        self.backbone = GRUODEBayesBackbone(
            input_size=configs.enc_in,
            hidden_size=configs.d_model,
            p_hidden=configs.gru_ode_bayes_p_hidden,
            prep_hidden=configs.gru_ode_bayes_prep_hidden,
            mixing=configs.gru_ode_bayes_mixing,
            solver=configs.gru_ode_bayes_solver,
            step_size=configs.gru_ode_bayes_step_size,
            dropout=configs.dropout,
        )
        self.classifier = nn.Linear(configs.d_model, configs.n_classes)

    @property
    def nfe(self) -> int:
        """Number of ODE vector-field evaluations in the latest forward pass."""
        return self.backbone.nfe

    @staticmethod
    def _default_times(values: Tensor) -> Tensor:
        length = values.size(1)
        return repeat(
            torch.arange(length, dtype=values.dtype, device=values.device)
            / max(length, 1),
            "l -> b l 1",
            b=values.size(0),
        )

    @staticmethod
    def _sanitize_times(times: Tensor, valid_rows: Tensor) -> Tensor:
        """Make padded timestamps constant and real timestamps monotone."""
        sanitized = []
        previous = times.new_zeros(times.size(0))
        for index in range(times.size(1)):
            current = torch.maximum(times[:, index, 0], previous)
            current = torch.where(valid_rows[:, index], current, previous)
            sanitized.append(current)
            previous = current
        if not sanitized:
            return times.new_empty(times.size(0), 0)
        return torch.stack(sanitized, dim=1)

    def forward(
        self,
        x: Tensor,
        x_mark: Tensor | None = None,
        x_mask: Tensor | None = None,
        y: Tensor | None = None,
        y_mark: Tensor | None = None,
        y_mask: Tensor | None = None,
        y_class: Tensor | None = None,
        exp_stage: str = "train",
        **kwargs,
    ) -> dict[str, Tensor]:
        batch_size, _, n_features = x.shape
        if x_mark is None:
            x_mark = self._default_times(x)
        if x_mask is None:
            x_mask = torch.ones_like(x)
        if y is None:
            if self.configs.task_name != "classification":
                logger.warning(
                    "y is missing. This is only expected during FLOP testing."
                )
            y = x.new_zeros(batch_size, self.pred_len, n_features)
        if y_mark is None:
            start = x_mark[:, -1:, :] if x_mark.size(1) else x.new_zeros(
                batch_size, 1, 1
            )
            increments = repeat(
                torch.arange(
                    1,
                    y.size(1) + 1,
                    dtype=x.dtype,
                    device=x.device,
                )
                / max(y.size(1), 1),
                "l -> b l 1",
                b=batch_size,
            )
            y_mark = start + increments
        if y_mask is None:
            y_mask = torch.ones_like(y)

        x_valid = x_mask.bool().any(dim=-1)
        x_times = self._sanitize_times(x_mark, x_valid)

        if self.configs.task_name in {
            "short_term_forecast",
            "long_term_forecast",
        }:
            y_valid = y_mask.bool().any(dim=-1)
            y_times = self._sanitize_times(y_mark, y_valid)
            mean, logvar, loss = self.backbone(
                context_values=x,
                context_times=x_times,
                context_mask=x_mask,
                target_times=y_times,
                target_values=y,
                target_mask=y_mask,
            )
            f_dim = -1 if self.configs.features == "MS" else 0
            return {
                "pred": mean[:, :, f_dim:],
                "pred_logvar": logvar[:, :, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:],
                "loss": loss,
            }

        if self.configs.task_name == "imputation":
            means = []
            logvars = []
            h = self.backbone.initial_hidden.unsqueeze(0).expand(batch_size, -1)
            prediction = self.backbone.p_model(h)
            previous_time = x_times.new_zeros(batch_size)
            total_nll = x.new_zeros(())
            total_kl = x.new_zeros(())
            n_targets = x.new_zeros(())
            n_observations = x.new_zeros(())
            self.backbone.reset_nfe()
            for index in range(x.size(1)):
                delta_t = (x_times[:, index] - previous_time).clamp_min(0.0)
                h, prediction = self.backbone._propagate(
                    h, prediction, delta_t
                )
                prior_mean, prior_logvar = prediction.chunk(2, dim=-1)
                means.append(prior_mean)
                logvars.append(prior_logvar.clamp(-10.0, 10.0))
                target_mask = y_mask[:, index].to(x.dtype)
                total_nll = total_nll + self.backbone.gru_obs.gaussian_nll(
                    prediction, y[:, index], target_mask
                ).sum()
                n_targets = n_targets + target_mask.sum()
                h, _ = self.backbone.gru_obs(
                    h, prediction, x[:, index], x_mask[:, index]
                )
                prediction = self.backbone.p_model(h)
                observation_mask = x_mask[:, index].to(x.dtype)
                total_kl = total_kl + self.backbone._post_update_kl(
                    prediction, x[:, index], observation_mask
                ).sum()
                n_observations = n_observations + observation_mask.sum()
                previous_time = torch.maximum(
                    previous_time, x_times[:, index]
                )
            mean = torch.stack(means, dim=1)
            logvar = torch.stack(logvars, dim=1)
            loss = total_nll / n_targets.clamp_min(1.0)
            loss = loss + self.backbone.mixing * (
                total_kl / n_observations.clamp_min(1.0)
            )
            f_dim = -1 if self.configs.features == "MS" else 0
            return {
                "pred": mean[:, :, f_dim:],
                "pred_logvar": logvar[:, :, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:],
                "loss": loss,
            }

        if self.configs.task_name == "classification":
            # Classification follows the reference model's terminal hidden
            # readout after alternating propagation and observation updates.
            h = self.backbone.initial_hidden.unsqueeze(0).expand(batch_size, -1)
            prediction = self.backbone.p_model(h)
            previous_time = x_times.new_zeros(batch_size)
            self.backbone.reset_nfe()
            for index in range(x.size(1)):
                delta_t = (x_times[:, index] - previous_time).clamp_min(0.0)
                h, prediction = self.backbone._propagate(
                    h, prediction, delta_t
                )
                h, _ = self.backbone.gru_obs(
                    h, prediction, x[:, index], x_mask[:, index]
                )
                prediction = self.backbone.p_model(h)
                previous_time = torch.maximum(
                    previous_time, x_times[:, index]
                )
            if y_class is None:
                y_class = x.new_zeros(batch_size, self.configs.n_classes)
            return {
                "pred_class": self.classifier(h),
                "true_class": y_class,
            }

        raise NotImplementedError(
            f"GRU-ODE-Bayes does not support {self.configs.task_name!r}"
        )
