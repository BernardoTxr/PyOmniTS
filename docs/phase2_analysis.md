# Phase 2 experiment metrics and figures

The metric recorder and chart tooling were implemented by GPT 5.6 SOL Extra
High Model; no external implementation was copied.

Normal PyOmniTS runs now save the efficiency and reproducibility information
needed by the analyses in `future_plans.md`. No separate benchmark invocation is
required. The existing result hierarchy and `metric.json` files are unchanged.

Each training repetition writes `run_metrics.json` in its `iterN` directory.
Each `eval_*` directory receives a self-contained evaluation snapshot with:

- dataset, model, task, horizon, missingness mechanism/rate, repetition, seed,
  full configuration, host, software versions, and GPU name;
- total/trainable parameter counts and parameter-plus-buffer size;
- wall-clock milliseconds per training iteration, measured over complete
  training epochs;
- peak CUDA memory allocated and reserved during training and evaluation;
- steady-state inference milliseconds per batch and per sample (the first
  batch is retained but excluded from the steady-state estimate);
- NFE per forward and milliseconds per NFE when the model exposes `model.nfe`;
- all normal evaluation metrics, including MSE and MAE.

Writes are atomic, only the main distributed process writes a record, and a
recording failure is logged without aborting an experiment. Set
`--save_run_metrics 0` to opt out. Use `--experiment_tag phase2` to label a
campaign and `--missing_mechanism` to distinguish `native`, `random`,
`timestamp_jitter`, `block_gap`, or `asynchronous_channels`. With the default
`auto`, a positive `missing_rate` is recorded as random injected missingness;
zero is recorded as native missingness.

## Publication figures

Generate the full figure set after or during a campaign:

```bash
python scripts/analysis/plot_phase2.py \
  --results-root storage/results \
  --output-dir storage/analysis/phase2
```

The command writes tidy per-run and aggregated CSV files, a JSON manifest, and
300-DPI PNG plus vector PDF figures:

- model MSE with 95% confidence intervals;
- MSE versus training time per iteration (GPU-memory bubble area);
- MSE versus time per NFE;
- MSE versus trainable parameter count;
- robustness versus missing rate;
- sensitivity versus forecast horizon.

Comparisons are grouped by dataset, context length, horizon, missingness
mechanism, and missingness rate, so unrelated experimental conditions are
never averaged in the same comparison panel. Repeated evaluations of one
checkpoint are de-duplicated by default. The collector also reads legacy
`metric.json` plus `configs.yaml` results and, when present, the older JSON
files under `metrics/`.

Useful filters include `--datasets`, `--models`, `--seq-len`, `--pred-len`,
`--missing-mechanisms`, `--batch-size`, `--hostname`, and `--gpu-name`.
Efficiency panels are automatically split by batch size, host, and GPU so the
script cannot silently average incomparable timing or memory measurements. A
chart is intentionally skipped when its required measurements or scenario
variation are unavailable; `manifest.json` records which outputs were skipped.

For fair timing figures, compare runs made on the same GPU, batch size, data
loader settings, and precision mode. Host and GPU metadata are included in the
CSV so mixed-hardware campaigns can be filtered before publication.
