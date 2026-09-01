# Code from: Bernardo Teixeira
# Created by Bernardo Teixeira <bernardoteixeira@usp.br>
# License: BSD-3-Clause

"""PyOmniTS forecasting adaptor for DTAMI-C.

The core transition is ported from
``ASOC2026-InterpolateInMTS/src/models/dtami_c.py``.  DTAMI-C shares the
observation ingestion and causal hidden-state decomposition used by
DTAMI-CIRC, but learns the orthogonal basis in which paired complex
eigenvalues perform rotation and scaling.
"""

import torch
from torch import Tensor

from models.DTAMI_CIRC import Model as DTAMICIRCModel
from models.DTAMI_CIRC import init_eigenvalues
from utils.ExpConfigs import ExpConfigs


class Model(DTAMICIRCModel):
    """DTAMI with complex eigenvalue pairs and a learnable orthogonal basis."""

    def __init__(self, configs: ExpConfigs):
        super().__init__(configs)

        self.beta = configs.dtami_beta

        # DTAMI-C has H/2 unconstrained complex-conjugate pairs, unlike the
        # real-FFT representation in DTAMI-CIRC, whose DC and Nyquist modes
        # must stay real.
        del self.sigma
        del self.omega

        self.U = torch.nn.Parameter(
            torch.empty(self.hidden_units, self.hidden_units)
        )
        torch.nn.init.orthogonal_(self.U)

        log_r, theta = init_eigenvalues(
            self.init_eigenvalues_config, self.hidden_units // 2
        )
        self.log_r = torch.nn.Parameter(log_r)
        self.theta = torch.nn.Parameter(theta)
        self.orthogonality_error: Tensor | None = None

    def load_state_dict(self, state_dict, strict=True):
        """Load current checkpoints and migrate the legacy ``log_theta`` key."""
        state_dict = dict(state_dict)
        if "log_theta" in state_dict and "theta" not in state_dict:
            state_dict["theta"] = state_dict.pop("log_theta").exp()
        return super().load_state_dict(state_dict, strict=strict)

    def parseval_loss(self) -> Tensor:
        """Soft orthogonality penalty used by the original DTAMI-C model."""
        identity = torch.eye(
            self.hidden_units, device=self.U.device, dtype=self.U.dtype
        )
        error = self.U.transpose(0, 1) @ self.U - identity
        return (self.beta / 2.0) * torch.linalg.matrix_norm(error).square()

    def post_step(self) -> None:
        """Apply the original Parseval update after each optimizer step."""
        if self.beta == 0:
            return
        with torch.no_grad():
            self.U.copy_(
                (1.0 + self.beta) * self.U
                - self.beta * ((self.U @ self.U.transpose(0, 1)) @ self.U)
            )
            identity = torch.eye(
                self.hidden_units, device=self.U.device, dtype=self.U.dtype
            )
            self.orthogonality_error = torch.linalg.matrix_norm(
                self.U.transpose(0, 1) @ self.U - identity
            ).detach()

    def time_translation(self, h_t: Tensor, deltat: Tensor) -> Tensor:
        """Translate state with learned-basis rotation-scaling blocks.

        This is algebraically equivalent to ``U @ D(dt) @ U.T @ h`` from
        the personal implementation, without allocating a dense ``[B,H,H]``
        block-diagonal tensor for every time step.
        """
        basis_state = h_t @ self.U
        even_state = basis_state[:, 0::2]
        odd_state = basis_state[:, 1::2]

        scale = torch.exp(self.log_r.unsqueeze(0) * deltat.unsqueeze(1))
        angle = self.theta.unsqueeze(0) * deltat.unsqueeze(1)
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)

        translated = torch.empty_like(basis_state)
        translated[:, 0::2] = scale * (
            cos_angle * even_state - sin_angle * odd_state
        )
        translated[:, 1::2] = scale * (
            sin_angle * even_state + cos_angle * odd_state
        )
        return translated @ self.U.transpose(0, 1)

    def forward(self, *args, **kwargs) -> dict:
        outputs = super().forward(*args, **kwargs)
        identity = torch.eye(
            self.hidden_units, device=self.U.device, dtype=self.U.dtype
        )
        self.orthogonality_error = torch.linalg.matrix_norm(
            self.U.transpose(0, 1) @ self.U - identity
        ).detach()
        outputs["regularization_loss"] = self.parseval_loss()
        return outputs
