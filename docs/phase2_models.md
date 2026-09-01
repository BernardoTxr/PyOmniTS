# Phase 2 irregular-time baselines

All seven baselines requested in `ASOC2026-InterpolateInMTS/future_plans.md`
are available through PyOmniTS's standard `models.<model_name>.Model` API.

| Baseline | PyOmniTS model name | Implementation source |
|---|---|---|
| Latent ODE | `Latent_ODE` | <https://github.com/YuliaRubanova/latent_ode> |
| GRU-ODE-Bayes | `GRU_ODE_Bayes` | <https://github.com/edebrouwer/gru_ode_bayes> |
| Neural CDE | `Neural_CDE` | <https://github.com/patrick-kidger/NeuralCDE> and <https://github.com/patrick-kidger/torchcde> |
| mTAN | `mTAN` | <https://github.com/reml-lab/mTAN> |
| Raindrop | `Raindrop` | <https://github.com/WenjieDu/PyPOTS> |
| CRU | `CRU` | <https://github.com/boschresearch/Continuous-Recurrent-Units> |
| GraFITi | `GraFITi` | <https://github.com/yalavarthivk/GraFITi> |

GRU-ODE-Bayes and Neural CDE were added for Phase 2. Both accept per-sample
irregular timestamps and per-channel observation masks, support forecasting,
imputation, and classification, and expose the latest vector-field evaluation
count as `model.nfe`. Forecast targets never enter either model's context
state.

## Reproducible launch scripts

Both new models have launch scripts for the five Phase 2 datasets currently
available in PyOmniTS: P12, USHCN, HumanActivity, MIMIC-III, and MIMIC-IV. For
example:

```bash
sh scripts/GRU_ODE_Bayes/USHCN.sh
sh scripts/Neural_CDE/USHCN.sh
```

MuJoCo Hopper is named in the Phase 2 plan but does not yet have a PyOmniTS
dataset adapter, so no Hopper launch script is included here.

## Verification

The focused tests check the reference GRU-ODE equations, Neural CDE control
construction, masked-value invariance, forecast-target causality, gradients,
configuration parsing, and learning on an irregular exponential-decay system:

```bash
python -m pytest -q \
  tests/models/test_GRU_ODE_Bayes.py \
  tests/models/test_Neural_CDE.py
```

With the fixed test seeds, GRU-ODE-Bayes reduced held-out MSE from
`0.21834119` to `0.00028232`, and Neural CDE reduced it from `0.38175830` to
`0.00375808`. A real USHCN batch (`x: [4, 287, 5]`, `y: [4, 3, 5]`) also
completed finite forward and backward passes for both models.
