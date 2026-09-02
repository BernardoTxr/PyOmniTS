#!/usr/bin/env bash
set -euo pipefail

# Code from: GPT 5.6 SOL Extra High Model
# Created for the DTAMI Phase 2.1 USHCN campaign.
# Baseline hyperparameters follow Appendix A.4 of HyperIMTS:
# https://arxiv.org/abs/2505.17431

PYTHON_BIN="${PYTHON_BIN:-python}"
MODELS="${MODELS:-DTAMI_C DTAMI_CIRC HyperIMTS GraFITi tPatchGNN Warpformer GNeuralFlow GRU_D}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-300}"
PATIENCE="${PATIENCE:-10}"
ITR="${ITR:-5}"
NUM_WORKERS="${NUM_WORKERS:-10}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-phase2_1_ushcn_hyperimts_protocol}"
MODEL_ID_PREFIX="${MODEL_ID_PREFIX:-phase2_1_ushcn}"
CHECKPOINTS="${CHECKPOINTS:-storage/results/}"

for model_name in $MODELS; do
    model_options=()
    case "$model_name" in
        DTAMI_C)
            model_options=(
                --collate_fn collate_fn
                --loss MSEWithRegularization
                --d_model 64
                --dtami_beta 0.5
                --dtami_h_ablation only_left
                --dtami_hidden_units 64
                --dtami_initializer_type uniform
                --dtami_n_traverse 1
                --learning_rate 1e-3
            )
            ;;
        DTAMI_CIRC)
            model_options=(
                --collate_fn collate_fn
                --loss MSE
                --d_model 64
                --dtami_h_ablation only_left
                --dtami_hidden_units 64
                --dtami_initializer_type uniform
                --dtami_n_traverse 1
                --learning_rate 1e-3
            )
            ;;
        HyperIMTS)
            model_options=(
                --collate_fn collate_fn
                --loss MSE
                --d_model 256
                --n_layers 1
                --n_heads 1
                --learning_rate 1e-3
            )
            ;;
        GraFITi)
            model_options=(
                --collate_fn collate_fn
                --loss MSE
                --d_model 128
                --n_layers 2
                --n_heads 4
                --learning_rate 1e-3
            )
            ;;
        tPatchGNN)
            model_options=(
                --collate_fn collate_fn_patch
                --loss MSE
                --d_model 32
                --dropout 0.5
                --n_heads 1
                --n_layers 3
                --node_dim 10
                --patch_len 10
                --learning_rate 1e-3
            )
            ;;
        Warpformer)
            model_options=(
                --collate_fn collate_fn
                --loss MSE
                --d_model 64
                --dropout 0.0
                --n_heads 1
                --n_layers 3
                --learning_rate 1e-3
            )
            ;;
        GNeuralFlow)
            model_options=(
                --collate_fn collate_fn
                --loss ModelProvidedLoss
                --neuralflows_time_net TimeTanh
                --neuralflows_flow_model resnet
                --neuralflows_flow_layers 2
                --neuralflows_latents 20
                --neuralflows_time_hidden_dim 8
                --hidden_layers 3
                --d_model 50
                --latent_ode_rec_dims 40
                --latent_ode_rec_layers 3
                --latent_ode_gen_layers 3
                --latent_ode_units 50
                --latent_ode_gru_units 50
                --learning_rate 1e-3
            )
            ;;
        GRU_D)
            model_options=(
                --collate_fn collate_fn
                --loss MSE
                --d_model 512
                --learning_rate 1e-3
            )
            ;;
        *)
            echo "Unsupported Phase 2.1 model: $model_name" >&2
            exit 2
            ;;
    esac

    "$PYTHON_BIN" main.py \
        --is_training 1 \
        --task_name short_term_forecast \
        --experiment_tag "$EXPERIMENT_TAG" \
        --checkpoints "$CHECKPOINTS" \
        --model_name "$model_name" \
        --model_id "${MODEL_ID_PREFIX}_${model_name}" \
        --dataset_name USHCN \
        --dataset_id USHCN \
        --dataset_root_path storage/datasets/USHCN \
        --features M \
        --seq_len 150 \
        --pred_len 3 \
        --enc_in 5 \
        --dec_in 5 \
        --c_out 5 \
        --missing_rate 0 \
        --missing_mechanism native \
        --train_epochs "$TRAIN_EPOCHS" \
        --patience "$PATIENCE" \
        --val_interval 1 \
        --itr "$ITR" \
        --batch_size 16 \
        --num_workers "$NUM_WORKERS" \
        --lr_scheduler DelayedStepDecayLR \
        --use_gpu 1 \
        --use_multi_gpu 0 \
        "${model_options[@]}"
done
