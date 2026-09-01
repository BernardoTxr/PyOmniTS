# Code from: https://github.com/Ladbaby/PyOmniTS
# Shared reproducible runner for GRU-ODE-Bayes and Neural CDE.

use_multi_gpu=0
if [ "$use_multi_gpu" -eq 0 ]; then
    launch_command="python"
else
    launch_command="accelerate launch"
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
    MIMIC_III)
        seq_len=72
        pred_len=3
        batch_size=32
        ;;
    MIMIC_IV)
        seq_len=2160
        pred_len=3
        batch_size=32
        ;;
    *)
        echo "Unsupported irregular dataset: $dataset_name" >&2
        exit 1
        ;;
esac

case "$model_name" in
    GRU_ODE_Bayes)
        loss_name="ModelProvidedLoss"
        d_model=100
        model_options="--gru_ode_bayes_mixing 1e-4 --gru_ode_bayes_p_hidden 25 --gru_ode_bayes_prep_hidden 10 --gru_ode_bayes_solver euler --gru_ode_bayes_step_size 0.05"
        ;;
    Neural_CDE)
        loss_name="MSE"
        d_model=64
        model_options="--neural_cde_adjoint 0 --neural_cde_hidden_layers 1 --neural_cde_hidden_width 128 --neural_cde_solver rk4"
        ;;
    *)
        echo "Unsupported model: $model_name" >&2
        exit 1
        ;;
esac

$launch_command main.py \
    --is_training 1 \
    --collate_fn "collate_fn" \
    --loss "$loss_name" \
    --d_model "$d_model" \
    $model_options \
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
    --train_epochs 300 \
    --patience 10 \
    --val_interval 1 \
    --itr 5 \
    --batch_size "$batch_size" \
    --learning_rate 1e-3

