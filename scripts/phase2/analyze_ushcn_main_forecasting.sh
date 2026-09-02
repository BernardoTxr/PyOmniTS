#!/usr/bin/env bash
# Code from: https://github.com/Ladbaby/PyOmniTS
# USHCN Phase 2.1/2.5 reporting implemented by GPT 5.6 SOL Extra High Model.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_ROOT="${RESULTS_ROOT:-storage/results}"
OUTPUT_DIR="${OUTPUT_DIR:-storage/analysis/phase2_1_ushcn}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-phase2_1_ushcn_hyperimts_protocol}"
MODELS=(DTAMI_C DTAMI_CIRC HyperIMTS GraFITi tPatchGNN Warpformer GNeuralFlow GRU_D)

"$PYTHON_BIN" scripts/analysis/table_phase2.py \
  --results-root "$RESULTS_ROOT" \
  --metrics-root metrics \
  --output-dir "$OUTPUT_DIR" \
  --dataset USHCN \
  --models "${MODELS[@]}" \
  --seq-len 150 \
  --pred-len 3 \
  --experiment-tag "$EXPERIMENT_TAG"

"$PYTHON_BIN" scripts/analysis/plot_phase2.py \
  --results-root "$RESULTS_ROOT" \
  --metrics-root metrics \
  --output-dir "$OUTPUT_DIR" \
  --datasets USHCN \
  --models "${MODELS[@]}" \
  --seq-len 150 \
  --pred-len 3 \
  --missing-mechanisms native \
  --batch-size 16 \
  --experiment-tag "$EXPERIMENT_TAG" \
  --formats pdf png
