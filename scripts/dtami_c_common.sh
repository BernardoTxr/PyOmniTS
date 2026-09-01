# Code from: Bernardo Teixeira
# Created by Bernardo Teixeira <bernardoteixeira@usp.br>
# License: BSD-3-Clause

# Shared Phase 2.1 runner for DTAMI-C on the irregular datasets. Dataset-named
# wrappers source this file so their file and directory names select the dataset
# and model in the same way as the other PyOmniTS launchers.

PYTHON_BIN="${PYTHON_BIN:-python}"
use_multi_gpu="${USE_MULTI_GPU:-0}"
train_epochs="${TRAIN_EPOCHS:-300}"
patience="${PATIENCE:-10}"
iterations="${ITR:-5}"
num_workers="${NUM_WORKERS:-10}"
use_gpu="${USE_GPU:-1}"
experiment_tag="${EXPERIMENT_TAG:-phase2_1_main_forecasting}"

if [ "$use_multi_gpu" -eq 0 ]; then
    launch_command=("$PYTHON_BIN")
else
    launch_command=(accelerate launch)
fi

script_file=$(readlink -f "$0")
model_dir=$(dirname "$script_file")
. "$model_dir/../globals.sh"

dataset_name=$(basename "$script_file" .sh)
dataset_subset_name=""
dataset_id=$dataset_name
get_dataset_info "$dataset_name" "$dataset_subset_name" || exit 1

model_name=$(basename "$model_dir")
model_id=$model_name

case "$dataset_name" in
    P12)
        seq_len=36
        pred_len=3
        batch_size=32
        ;;
    USHCN)
        seq_len=150
        pred_len=3
        batch_size=16
        ;;
    HumanActivity)
        seq_len=3000
        pred_len=300
        batch_size=32
        ;;
    *)
        echo "Unsupported Phase 2.1 dataset: $dataset_name" >&2
        exit 1
        ;;
esac

"${launch_command[@]}" main.py \
    --is_training 1 \
    --experiment_tag "$experiment_tag" \
    --collate_fn collate_fn \
    --loss MSEWithRegularization \
    --d_model 64 \
    --dtami_beta 0.5 \
    --dtami_h_ablation only_left \
    --dtami_hidden_units 64 \
    --dtami_initializer_type uniform \
    --dtami_n_traverse 1 \
    --use_gpu "$use_gpu" \
    --use_multi_gpu "$use_multi_gpu" \
    --dataset_root_path "$dataset_root_path" \
    --model_id "$model_id" \
    --model_name "$model_name" \
    --dataset_name "$dataset_name" \
    --dataset_id "$dataset_id" \
    --features M \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --enc_in "$n_variables" \
    --dec_in "$n_variables" \
    --c_out "$n_variables" \
    --train_epochs "$train_epochs" \
    --patience "$patience" \
    --val_interval 1 \
    --itr "$iterations" \
    --batch_size "$batch_size" \
    --num_workers "$num_workers" \
    --learning_rate 1e-3
