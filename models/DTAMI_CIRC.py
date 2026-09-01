# Code from: Bernardo Teixeira
# Created by Bernardo Teixeira <bernardoteixeira@usp.br>
# License: BSD-3-Clause

import torch
import torch.nn as nn
from einops import repeat
from torch import Tensor

from utils.ExpConfigs import ExpConfigs
from utils.globals import logger

# (use_local, use_left, use_right) per ablation.
# Ported from ASOC2026-InterpolateInMTS/src/models/dtami.py
_VALID_ABLATIONS = {
    "full":        (True,  True,  True),
    "only_local":  (True,  False, False),
    "only_left":   (False, True,  False),
    "only_right":  (False, False, True),
    "no_local":    (False, True,  True),
    "no_left":     (True,  False, True),
    "no_right":    (True,  True,  False),
}


class Model(nn.Module):
    """
    - Paper: not submitted yet

    DTAMI-CIRC: DTAMI with a continuous circulant time-translation operator.
    Replaces the learnable orthogonal matrix U with the fixed DFT basis F,
    giving an O(H) parameterization of the time-translation operator:

        g_θ(h_t, Δt) = F⁻¹ diag(exp(λ·Δt)) F h_t
                     = irfft( rfft(h_t) ⊙ exp(λ·Δt) )

    where λ_k = σ_k + iω_k. Core algorithm ported verbatim from
    ASOC2026-InterpolateInMTS/src/models/dtami_circ.py.

    Currently only short_term_forecast / long_term_forecast are supported.
    """
    def __init__(self, configs: ExpConfigs):
        super().__init__()
        if configs.task_name not in ["short_term_forecast", "long_term_forecast"]:
            raise NotImplementedError(f"{configs.task_name} not implemented for DTAMI_CIRC (forecast tasks only)")
        # BEGIN adaptor
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        n_features = configs.enc_in
        # END adaptor

        self.configs = configs
        self.n_features = n_features

        hidden_units = configs.dtami_hidden_units
        n_traverse = configs.dtami_n_traverse
        h_ablation = configs.dtami_h_ablation
        init_eigenvalues_config = {
            "type": configs.dtami_initializer_type,
            "r_min": configs.dtami_init_r_min,
            "r_max": configs.dtami_init_r_max,
            "f_min": configs.dtami_init_f_min,
            "f_max": configs.dtami_init_f_max,
            "r_value": configs.dtami_init_r_value,
            "theta_value": configs.dtami_init_theta_value,
        }

        if hidden_units % 2 != 0:
            raise ValueError(
                f"DTAMICIRC requires hidden_units to be even, got {hidden_units}"
            )
        if n_traverse < 1:
            raise ValueError(
                f"DTAMICIRC requires n_traverse >= 1, got {n_traverse}"
            )
        if h_ablation not in _VALID_ABLATIONS:
            raise ValueError(
                f"h_ablation must be one of {list(_VALID_ABLATIONS)}, got {h_ablation!r}"
            )
        self.hidden_units = hidden_units
        self.input_size = n_features
        self.n_traverse = n_traverse
        self.h_ablation = h_ablation
        self.init_eigenvalues_config = init_eigenvalues_config

        self.ingest_observation = nn.GRU(
            self.input_size * 2, self.hidden_units, num_layers=1, batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_units, self.input_size)

        k_free = self.hidden_units // 2 - 1
        if k_free > 0:
            _log_r, _theta = init_eigenvalues(init_eigenvalues_config, k_free)
            _sigma_init = torch.zeros(self.hidden_units // 2 + 1)
            _sigma_init[1:-1] = _log_r
        else:
            _sigma_init = torch.zeros(self.hidden_units // 2 + 1)
            _theta = torch.zeros(0)
        self.sigma = nn.Parameter(_sigma_init)
        self.omega = nn.Parameter(_theta)

    def incorporate_x(self, features, h_t):
        features = features.unsqueeze(dim=1)
        h_t = h_t.unsqueeze(dim=0)
        _, h_t_ = self.ingest_observation(features, h_t)
        return h_t_.squeeze(dim=0)

    def predict_output(self, h_t):
        return self.output_layer(h_t)

    def time_translation(self, h_t, deltat):
        """Translate hidden state by Δt via circular convolution in Fourier domain."""
        H = self.hidden_units

        omega_full = torch.nn.functional.pad(self.omega, (1, 1), value=0.0)  # [H//2+1]

        sigma_dt = self.sigma.unsqueeze(0) * deltat.unsqueeze(1)    # [B, H//2+1]
        omega_dt = omega_full.unsqueeze(0) * deltat.unsqueeze(1)    # [B, H//2+1]

        r_dt   = torch.exp(sigma_dt)
        cos_dt = torch.cos(omega_dt)
        sin_dt = torch.sin(omega_dt)
        c_freq = torch.complex(r_dt * cos_dt, r_dt * sin_dt)       # [B, H//2+1]

        h_freq = torch.fft.rfft(h_t, n=H, dim=-1)
        return torch.fft.irfft(h_freq * c_freq, n=H, dim=-1)

    def process_sequence(self, features, deltat, h_0, context_mask, return_components=False):
        # Forecasting uses only the causal left state. Avoid computing the
        # backward traversal when it will be discarded. Keep the full path for
        # component diagnostics so callers can still inspect all decompositions.
        if self.h_ablation == "only_left" and not return_components:
            batch_size = features.size(0)
            seq_len = features.size(1)
            h_left = torch.zeros(
                batch_size,
                seq_len,
                self.hidden_units,
                device=features.device,
                dtype=features.dtype,
            )
            h_left[:, 0, :] = h_0[:, 0, :]
            h_t = h_0[:, 0, :].clone()

            for step_id in range(deltat.size(-1)):
                step_context_mask = context_mask[:, step_id]
                h_t[step_context_mask] = self.incorporate_x(
                    features=features[:, step_id, :][step_context_mask],
                    h_t=h_t[step_context_mask],
                )
                h_t = self.time_translation(
                    h_t=h_t, deltat=deltat[:, step_id]
                )
                h_left[:, step_id + 1, :] = h_t

            return h_left

        steps = range(deltat.size(-1))

        h_0_star = h_0.clone()
        h_0_star[context_mask] = self.incorporate_x(
            features=features[context_mask],
            h_t=h_0[context_mask],
        )

        batch_size = features.size(0)
        seq_len = features.size(1)

        h_sides = torch.zeros(
            2 * batch_size, seq_len, self.hidden_units, device=features.device
        )
        h_sides[:batch_size, 0, :] = h_0[:, 0, :]
        h_sides[batch_size:, 0, :] = h_0[:, -1, :]

        h_t = torch.cat((h_0[:, 0, :].clone(), h_0[:, -1, :].clone()), dim=0)

        for step_id in steps:
            features_lbound = features[:, step_id, :]
            features_ubound = features[:, -(step_id + 1), :]
            step_features = torch.cat((features_lbound, features_ubound), dim=0)
            step_context_mask = torch.cat(
                (context_mask[:, step_id], context_mask[:, -(step_id + 1)]), dim=0
            )
            available_info = step_features[step_context_mask]
            h_t[step_context_mask] = self.incorporate_x(
                features=available_info, h_t=h_t[step_context_mask]
            )
            time_distance_lbound = deltat[:, step_id]
            time_distance_ubound = -deltat[:, -(step_id + 1)]
            step_deltat = torch.cat((time_distance_lbound, time_distance_ubound), dim=0)
            h_t = self.time_translation(h_t=h_t, deltat=step_deltat)
            h_sides[:, step_id + 1, :] = h_t

        h_local = h_0_star
        h_left  = h_sides[:batch_size]
        h_right = torch.flip(h_sides[batch_size:], dims=[1])

        use_local, use_left, use_right = _VALID_ABLATIONS[self.h_ablation]
        active = (
            ([h_local] if use_local else [])
            + ([h_left]  if use_left  else [])
            + ([h_right] if use_right else [])
        )
        h_out = sum(active) / len(active)

        if return_components:
            return h_out, h_local, h_left, h_right
        return h_out

    def _dtamicirc_forward(self, data, timestamps, h_0, context_mask, return_components=False):
        """Core DTAMI-CIRC forward pass. Ported from ASOC2026's DTAMICIRC.forward;
        renamed since the outer nn.Module entry point must be named `forward`
        and follow PyOmniTS's (x, x_mark, x_mask, y, y_mark, y_mask, ...) contract.
        """
        time_distances = torch.diff(timestamps, dim=1)
        h_n = h_0

        h_local_history, h_left_history, h_right_history = [], [], []

        for _ in range(self.n_traverse):
            result = self.process_sequence(
                h_0=h_n,
                features=data,
                deltat=time_distances,
                context_mask=context_mask,
                return_components=return_components,
            )
            if return_components:
                h_n, h_local, h_left, h_right = result
                h_local_history.append(h_local)
                h_left_history.append(h_left)
                h_right_history.append(h_right)
            else:
                h_n = result

        out_block = self.predict_output(h_n)

        if return_components:
            return out_block, (
                torch.stack(h_local_history, dim=0),
                torch.stack(h_left_history, dim=0),
                torch.stack(h_right_history, dim=0),
            )
        return out_block

    def forward(
        self, 
        x: Tensor,
        x_mark: Tensor | None = None, 
        x_mask: Tensor | None = None, 
        y: Tensor | None = None, 
        y_mark: Tensor | None = None, 
        y_mask: Tensor | None = None, 
        y_class: Tensor | None = None, 
        **kwargs
    ) -> dict:
        # BEGIN adaptor
        BATCH_SIZE, SEQ_LEN, ENC_IN = x.shape
        Y_LEN = self.pred_len
        if x_mark is None:
            x_mark = repeat(torch.arange(end=x.shape[1], dtype=x.dtype, device=x.device) / x.shape[1], "L -> B L 1", B=x.shape[0])
        if x_mask is None:
            x_mask = torch.ones_like(x, device=x.device, dtype=x.dtype)
        if y is None:
            if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
                logger.warning(f"y is missing for the model input. This is only reasonable when the model is testing flops!")
            y = torch.ones((BATCH_SIZE, Y_LEN, ENC_IN), dtype=x.dtype, device=x.device)
        if y_mark is None:
            y_mark = repeat(torch.arange(end=y.shape[1], dtype=y.dtype, device=y.device) / y.shape[1], "L -> B L 1", B=y.shape[0])
        if y_mask is None:
            y_mask = torch.ones_like(y, device=y.device, dtype=y.dtype)
        # END adaptor

        if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
            PRED_LEN = y.shape[1]

            X = torch.cat([x, torch.zeros_like(y)], dim=1)                     # [B, L, ENC_IN]
            missing_mask = torch.cat([x_mask, torch.zeros_like(y_mask)], dim=1) # [B, L, ENC_IN]
            data = torch.cat([X, missing_mask], dim=-1)                       # [B, L, 2*ENC_IN]
            timestamps = torch.cat([x_mark, y_mark], dim=1)[:, :, 0]          # [B, L]

            # A time point is context only when at least one input variable is
            # observed. This excludes padded history as well as the forecast
            # horizon while retaining the per-variable mask in ``data``.
            observed_history = x_mask.to(dtype=torch.bool).any(dim=-1)         # [B, SEQ_LEN]
            future_context = torch.zeros(
                BATCH_SIZE, PRED_LEN, device=x.device, dtype=torch.bool
            )
            context_mask = torch.cat([observed_history, future_context], dim=1) # [B, L]

            L = context_mask.size(1)

            h_0 = torch.zeros(BATCH_SIZE, L, self.hidden_units, device=x.device, dtype=x.dtype)

            out_block = self._dtamicirc_forward(data, timestamps, h_0, context_mask) # [B, L, ENC_IN]

            f_dim = -1 if self.configs.features == 'MS' else 0
            return {
                "pred": out_block[:, -PRED_LEN:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:]
            }
        else:
            raise NotImplementedError(f"{self.configs.task_name} not implemented for DTAMI_CIRC")

"""Eigenvalue initializers for DTAMI-C.

Selected via the ``initializer_eigenvalues`` key in the model config:

    model:
      architecture: dtami_c
      initializer_eigenvalues:
        type: glorot-based
        r_min: 0.1
        r_max: 0.9

All initializers return ``(log_r, theta)`` tensors of shape ``[k]``,
where ``k = hidden_units // 2`` is the number of complex-conjugate pairs.
``log_r`` and ``theta`` are used directly as the initial values of the
corresponding ``nn.Parameter`` objects in ``DTAMIC``.

``theta`` is the angular frequency stored as an unconstrained real number
(matching the paper's direct parametrization ``Λ = diag(exp(-ν + iθ))``
where ``log_r = -ν`` and ``theta = θ``).

Available types
---------------
fixed-value
    All k pairs share the same magnitude r and angle θ.
    Params: ``r_value`` (float), ``theta_value`` (float).

uniform  *(default)*
    Magnitude r ~ Uniform(0, 1), angle θ ~ Uniform(0, 2π), independently
    for each pair.  No parameters required.

glorot-based
    Uniform distribution on the complex annulus
    {z ∈ ℂ : r_min ≤ |z| ≤ r_max} (Lemma 3.2).
    Params: ``r_min`` (float), ``r_max`` (float), both in [0, 1].

frequency-uniform
    Angular frequency θ = 2πf, where f ~ Uniform(f_min, f_max) cyc/time.
    Magnitude r sampled from the glorot annulus.
    Params: ``f_min`` (float, default 0.0), ``f_max`` (float, required),
            ``r_min`` (float, default 0.9), ``r_max`` (float, default 1.0).
    Use when the expected signal frequency range is roughly known.

frequency-log-uniform
    Angular frequency θ = 2πf, where log(f) ~ Uniform(log(f_min), log(f_max)).
    Gives equal probability mass per frequency decade — better when the
    frequency range spans more than one order of magnitude.
    Magnitude r sampled from the glorot annulus.
    Params: ``f_min`` (float > 0, default 0.5), ``f_max`` (float, required),
            ``r_min`` (float, default 0.9), ``r_max`` (float, default 1.0).
"""

import math
import torch

_EPS = 1e-8  # small floor to avoid log(0)


def init_eigenvalues(init_config: dict, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialise DTAMI-C eigenvalue parameters.

    Args:
        init_config: Dict with at least a ``type`` key.  Additional keys
                     depend on the chosen strategy.
        k:           Number of complex-conjugate pairs (``hidden_units // 2``).

    Returns:
        (log_r, theta) — each a float32 tensor of shape ``[k]``.
        ``log_r`` is the log-magnitude; ``theta`` is the angular frequency
        stored directly as an unconstrained real (no extra exp()).
    """
    init_type = str(init_config.get("type", "uniform")).lower()

    if init_type == "fixed-value":
        return _fixed_value(init_config, k)
    elif init_type == "uniform":
        return _uniform(k)
    elif init_type == "glorot-based":
        return _glorot_based(init_config, k)
    elif init_type == "frequency-uniform":
        return _frequency_uniform(init_config, k)
    elif init_type == "frequency-log-uniform":
        return _frequency_log_uniform(init_config, k)
    else:
        raise ValueError(
            f"Unknown initializer_eigenvalues type: {init_type!r}. "
            "Choose from: fixed-value, uniform, glorot-based, "
            "frequency-uniform, frequency-log-uniform"
        )


# ---------------------------------------------------------------------------
# Individual strategies
# ---------------------------------------------------------------------------

def _fixed_value(cfg: dict, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """All pairs share the same r and θ.

    Config keys:
        r_value     (float) — magnitude, must be > 0
        theta_value (float) — angular frequency in rad/time-unit, must be > 0
    """
    r_value = float(cfg["r_value"])
    theta_value = float(cfg["theta_value"])
    if r_value <= 0:
        raise ValueError(f"fixed-value: r_value must be > 0, got {r_value}")
    if theta_value <= 0:
        raise ValueError(f"fixed-value: theta_value must be > 0, got {theta_value}")
    log_r = torch.full((k,), math.log(r_value))
    theta = torch.full((k,), theta_value)
    return log_r, theta


def _uniform(k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """r ~ Uniform(0, 1), θ ~ Uniform(0, 2π), independently for each pair."""
    r = torch.rand(k).clamp(min=_EPS)              # (0, 1]
    theta = torch.rand(k) * 12.0 * math.pi           # [0, 12π) — stored directly
    return torch.log(r), theta


def _glorot_based(cfg: dict, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform distribution on the annulus {z : r_min ≤ |z| ≤ r_max}.

    Lemma 3.2:  let u1, u2 ~ Uniform(0, 1).
        v     = -1/2 · ln(u1·(r_max² - r_min²) + r_min²)
        r     = exp(-v)  =  sqrt(u1·(r_max² - r_min²) + r_min²)
        θ     = 2π · u2
    Then exp(-v + iθ) is uniformly distributed on the annulus.

    Config keys:
        r_min (float) — inner radius, in [0, 1]
        r_max (float) — outer radius, in [0, 1], must be ≥ r_min
    """
    r_min = float(cfg["r_min"])
    r_max = float(cfg["r_max"])
    if r_min < 0 or r_max < 0:
        raise ValueError(f"glorot-based: r_min and r_max must be >= 0, got {r_min}, {r_max}")
    if r_min > r_max:
        raise ValueError(f"glorot-based: r_min must be <= r_max, got {r_min} > {r_max}")

    u1 = torch.rand(k)
    u2 = torch.rand(k)

    # r² = u1·(r_max² - r_min²) + r_min²  →  log_r = 0.5·log(r²)
    r_sq = u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2
    log_r = 0.5 * torch.log(r_sq.clamp(min=_EPS))

    # θ = 2π·u2 — stored directly as unconstrained real
    theta = 2.0 * math.pi * u2

    return log_r, theta


def _frequency_uniform(cfg: dict, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """f ~ Uniform(f_min, f_max) cyc/time, θ = 2πf.

    Unlike glorot-based (which draws θ ∈ [0, 2π] and thus covers only
    f ∈ [0, 1] cyc/time), this initializer covers any desired frequency
    band and is appropriate for continuous-time data where Δt << 1.

    Config keys:
        f_min (float) — lower frequency bound in cyc/time (default 0.0).
        f_max (float) — upper frequency bound in cyc/time (required).
        r_min (float) — inner radius for glorot annulus (default 0.9).
        r_max (float) — outer radius for glorot annulus (default 1.0).
    """
    f_min = float(cfg.get("f_min", 0.0))
    f_max = float(cfg["f_max"])
    r_min = float(cfg.get("r_min", 0.9))
    r_max = float(cfg.get("r_max", 1.0))
    if f_min < 0:
        raise ValueError(f"frequency-uniform: f_min must be >= 0, got {f_min}")
    if f_max <= f_min:
        raise ValueError(f"frequency-uniform: f_max must be > f_min, got {f_max} <= {f_min}")
    if r_min > r_max:
        raise ValueError(f"frequency-uniform: r_min must be <= r_max, got {r_min} > {r_max}")

    f = torch.rand(k) * (f_max - f_min) + f_min   # [f_min, f_max]
    theta = 2.0 * math.pi * f                       # θ = 2πf (unbounded)

    u1 = torch.rand(k)
    r_sq = u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2
    log_r = 0.5 * torch.log(r_sq.clamp(min=_EPS))

    return log_r, theta


def _frequency_log_uniform(cfg: dict, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """log(f) ~ Uniform(log(f_min), log(f_max)), θ = 2πf.

    Gives equal probability mass per frequency decade.  Prefer this over
    frequency-uniform when the plausible frequency range spans more than
    one order of magnitude (e.g. f ∈ [0.5, 50] cyc/time).

    Config keys:
        f_min (float > 0) — lower frequency bound in cyc/time (default 0.5).
        f_max (float)     — upper frequency bound in cyc/time (required).
        r_min (float)     — inner radius for glorot annulus (default 0.9).
        r_max (float)     — outer radius for glorot annulus (default 1.0).
    """
    f_min = float(cfg.get("f_min", 0.5))
    f_max = float(cfg["f_max"])
    r_min = float(cfg.get("r_min", 0.9))
    r_max = float(cfg.get("r_max", 1.0))
    if f_min <= 0:
        raise ValueError(f"frequency-log-uniform: f_min must be > 0, got {f_min}")
    if f_max <= f_min:
        raise ValueError(f"frequency-log-uniform: f_max must be > f_min, got {f_max} <= {f_min}")
    if r_min > r_max:
        raise ValueError(f"frequency-log-uniform: r_min must be <= r_max, got {r_min} > {r_max}")

    log_f = torch.rand(k) * (math.log(f_max) - math.log(f_min)) + math.log(f_min)
    f = torch.exp(log_f)                            # f ∈ [f_min, f_max], log-uniform
    theta = 2.0 * math.pi * f                       # θ = 2πf (unbounded)

    u1 = torch.rand(k)
    r_sq = u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2
    log_r = 0.5 * torch.log(r_sq.clamp(min=_EPS))

    return log_r, theta
