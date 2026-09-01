# Code from: https://github.com/Ladbaby/PyOmniTS
# Code adapted from: https://github.com/patrick-kidger/NeuralCDE
# Copyright 2020 Patrick Kidger, James Morrill, James Foster, Terry Lyons
# Licensed under the Apache License, Version 2.0.

import torch
import torch.nn as nn
from einops import repeat
from torch import Tensor

from layers.Neural_CDE import NeuralCDEBackbone
from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


class Model(nn.Module):
    """
    - paper: "Neural Controlled Differential Equations for Irregular Time
      Series" (NeurIPS 2020 Spotlight)
    - paper link: https://papers.neurips.cc/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html
    - code adapted from: https://github.com/patrick-kidger/NeuralCDE
    - solver library: https://github.com/patrick-kidger/torchcde

    As in the reference preprocessing, the control contains time, cumulative
    observation intensity, and values. Values are causally carried forward so
    padded or missing entries cannot leak into the path. During forecasting,
    only the time channel changes after the context window.
    """

    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.configs = configs
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        self.backbone = NeuralCDEBackbone(
            n_features=configs.enc_in,
            hidden_channels=configs.d_model,
            hidden_hidden_channels=configs.neural_cde_hidden_width,
            num_hidden_layers=configs.neural_cde_hidden_layers,
            solver=configs.neural_cde_solver,
            adjoint=bool(configs.neural_cde_adjoint),
        )
        self.classifier = nn.Linear(configs.d_model, configs.n_classes)

    @property
    def nfe(self) -> int:
        """Number of CDE vector-field evaluations in the latest forward pass."""
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
        if y_mask is None:
            y_mask = torch.ones_like(y)

        x_valid = x_mask.bool().any(dim=-1)
        x_times = self._sanitize_times(x_mark, x_valid)

        if self.configs.task_name in {
            "short_term_forecast",
            "long_term_forecast",
        }:
            if y_mark is None:
                start = x_times[:, -1:] if x_times.size(1) else x.new_zeros(
                    batch_size, 1
                )
                increments = repeat(
                    torch.arange(
                        1,
                        y.size(1) + 1,
                        dtype=x.dtype,
                        device=x.device,
                    )
                    / max(y.size(1), 1),
                    "l -> b l",
                    b=batch_size,
                )
                y_times = start + increments
            else:
                y_valid = y_mask.bool().any(dim=-1)
                y_times = self._sanitize_times(y_mark, y_valid)

            target_values = x.new_zeros(y.shape)
            target_observations = x.new_zeros(y_mask.shape)
            values = torch.cat([x, target_values], dim=1)
            mask = torch.cat([x_mask, target_observations], dim=1)
            times = torch.cat([x_times, y_times], dim=1)
            control = self.backbone.build_control(values, times, mask)
            _, predictions = self.backbone(control)
            predictions = predictions[:, -y.size(1) :]

            f_dim = -1 if self.configs.features == "MS" else 0
            return {
                "pred": predictions[:, :, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:],
            }

        if self.configs.task_name == "imputation":
            control = self.backbone.build_control(x, x_times, x_mask)
            _, predictions = self.backbone(control)
            f_dim = -1 if self.configs.features == "MS" else 0
            return {
                "pred": predictions[:, :, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:],
            }

        if self.configs.task_name == "classification":
            control = self.backbone.build_control(x, x_times, x_mask)
            hidden, _ = self.backbone(control)
            if y_class is None:
                y_class = x.new_zeros(batch_size, self.configs.n_classes)
            return {
                "pred_class": self.classifier(hidden[:, -1]),
                "true_class": y_class,
            }

        raise NotImplementedError(
            f"Neural CDE does not support {self.configs.task_name!r}"
        )
