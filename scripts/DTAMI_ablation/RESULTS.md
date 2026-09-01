# DTAMI forecasting ablation

Exploratory seed-2024 results from the PyOmniTS forecasting task. All learned
models use the native dataset split and `collate_fn`, hidden/latent width 64,
Adam at 1e-3, a maximum of 300 epochs, patience 10, and masked evaluation.
DTAMI-C and DTAMI-CIRC use the causal `only_left` decomposition with one
traversal. DTAMI-C uses the stable Parseval setting `beta=0.5`.

Lower is better. Bold denotes the best learned model for a dataset.

| Dataset | Model | MSE | MAE |
|---|---|---:|---:|
| Human Activity (3000 -> 300) | DTAMI-C | 0.139991 | 0.284653 |
| | DTAMI-CIRC | 0.138859 | 0.283353 |
| | **mTAN** | **0.089510** | **0.215340** |
| | GRU-D | 0.092799 | 0.220663 |
| | Last observation (untrained) | 0.060507 | 0.122516 |
| PhysioNet 2012 (36 -> 3) | DTAMI-C | 0.355467 | 0.407223 |
| | DTAMI-CIRC | 0.361494 | 0.410443 |
| | mTAN | 0.370614 | 0.422201 |
| | **GRU-D** | **0.320224** | **0.380406** |
| | Last observation (untrained) | 0.408244 | 0.398385 |
| USHCN (150 -> 3) | DTAMI-C | 0.190153 | 0.298452 |
| | **DTAMI-CIRC** | **0.186993** | **0.287909** |
| | mTAN | 0.787372 | 0.559646 |
| | GRU-D | 0.212002 | 0.297383 |
| | Last observation (untrained) | 0.736384 | 0.395075 |

## Interpretation

- Human Activity is dominated by local continuity: the untrained last-value
  baseline beats every learned model. mTAN and corrected GRU-D are the best
  learned methods, while learning DTAMI-C's basis adds no value over CIRC.
- PhysioNet is a short, sparse clinical horizon. Corrected GRU-D's causal LOCF
  and feature-wise decay are the best bias. DTAMI-C's learned basis improves
  over the fixed Fourier basis by about 1.7% MSE, and both beat mTAN.
- USHCN has only 369 observed test targets. The compact fixed Fourier prior of
  CIRC is both the most accurate and much cheaper than learning a dense basis.
  mTAN is high-variance here: five repeated inference seeds on the same
  checkpoint gave MSE 0.717425 +/- 0.058110, still far behind both DTAMI models.

The GRU-D results above were produced after correcting the PyOmniTS adaptor's
mask semantics, causal LOCF, and feature-wise accumulated time gaps. Older
GRU-D result folders from before that correction are not comparable.

These are single-training-seed exploratory results, not publication estimates.
Run five seeds with:

```bash
ITR=5 scripts/DTAMI_ablation/forecasting.sh
```

The launcher also accepts `DATASETS`, `MODELS`, `DTAMI_BETA`, and
`DTAMI_N_TRAVERSE` environment overrides.
