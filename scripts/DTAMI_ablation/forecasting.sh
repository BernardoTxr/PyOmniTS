#!/usr/bin/env bash
set -euo pipefail

# Matched one- or multi-seed forecasting ablation. Override the space-separated
# DATASETS/MODELS variables, ITR, DTAMI_BETA, or DTAMI_N_TRAVERSE from the
# environment to run a subset or a DTAMI traversal sweep.
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASETS="${DATASETS:-HumanActivity P12 USHCN}"
MODELS="${MODELS:-DTAMI_C DTAMI_CIRC mTAN GRU_D}"
ITR="${ITR:-1}"
DTAMI_BETA="${DTAMI_BETA:-0.5}"
DTAMI_N_TRAVERSE="${DTAMI_N_TRAVERSE:-1}"

for dataset_name in ${DATASETS}; do
    case "${dataset_name}" in
        HumanActivity)
            dataset_root_path="storage/datasets/HumanActivity"
            seq_len=3000
            pred_len=300
            n_variables=12
            batch_size=32
            ;;
        P12)
            dataset_root_path="storage/datasets/P12"
            seq_len=36
            pred_len=3
            n_variables=36
            batch_size=32
            ;;
        USHCN)
            dataset_root_path="storage/datasets/USHCN"
            seq_len=150
            pred_len=3
            n_variables=5
            batch_size=16
            ;;
        *)
            echo "Unsupported dataset: ${dataset_name}" >&2
            exit 2
            ;;
    esac

    for model_name in ${MODELS}; do
        model_id="ablation_${model_name}"
        loss="MSE"
        case "${model_name}" in
            DTAMI_C)
                loss="MSEWithRegularization"
                ;;
            mTAN)
                loss="ModelProvidedLoss"
                ;;
            DTAMI_CIRC|GRU_D)
                ;;
            *)
                echo "Unsupported model: ${model_name}" >&2
                exit 2
                ;;
        esac

        "${PYTHON_BIN}" main.py \
            --is_training 1 \
            --model_name "${model_name}" \
            --model_id "${model_id}" \
            --dataset_name "${dataset_name}" \
            --dataset_id "${dataset_name}" \
            --dataset_root_path "${dataset_root_path}" \
            --collate_fn collate_fn \
            --features M \
            --seq_len "${seq_len}" \
            --pred_len "${pred_len}" \
            --enc_in "${n_variables}" \
            --dec_in "${n_variables}" \
            --c_out "${n_variables}" \
            --d_model 64 \
            --dtami_hidden_units 64 \
            --dtami_n_traverse "${DTAMI_N_TRAVERSE}" \
            --dtami_h_ablation only_left \
            --dtami_beta "${DTAMI_BETA}" \
            --dtami_initializer_type uniform \
            --loss "${loss}" \
            --learning_rate 1e-3 \
            --train_epochs 300 \
            --patience 10 \
            --val_interval 1 \
            --itr "${ITR}" \
            --batch_size "${batch_size}" \
            --num_workers 10 \
            --use_gpu 1 \
            --use_multi_gpu 0
    done
done
