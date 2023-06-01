#!/bin/bash
# shellcheck disable=SC2164
cd ../src

export PreTrainData_List=(geom2d_nmol500000_nconf1 geom2d_nmol500000_nconf1)
# export FineTuneData_List=(bbbp tox21 toxcast sider clintox muv hiv bace)
export PreTrainer_List=(AM)

# === Pre-Trainings: 2D GNN ===
for PreTrainData in "${PreTrainData_List[@]}"; do
    for PreTrainer in "${PreTrainer_List[@]}"; do
        python run_pretraining.py \
            --pretrainer="$PreTrainer" \
            --dataset="$PreTrainData" \
            --epochs=100 \
            --output_model_dir=./pretrain_models_hyperopt/;
    done
done

# === Pre-Trainings: 3D GNN ===


