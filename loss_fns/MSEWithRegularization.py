# Code from: Bernardo Teixeira
# License: BSD-3-Clause

"""Masked MSE plus a scalar regularizer supplied by the model."""

from torch import Tensor

from loss_fns.MSE import Loss as MSELoss


class Loss(MSELoss):
    def forward(
        self,
        pred: Tensor,
        true: Tensor,
        mask: Tensor | None = None,
        regularization_loss: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        result = super().forward(pred=pred, true=true, mask=mask, **kwargs)
        if regularization_loss is not None:
            result["loss"] = result["loss"] + regularization_loss
        return result
