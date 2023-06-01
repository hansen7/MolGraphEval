#!/bin/bash
# shellcheck disable=SC2164
cd ~/GraphEval_dev

export PreTrainData=zinc_standard_agent
export LearningRate=0.001
export PreTrainSeed=1
export PreTrainer=AM
export MaskRate=0.15

# python src/run_pretraining.py \
#     --project_name=$PreTrainer \
#     --pretrainer=$PreTrainer \
#     --dataset=$PreTrainData \
#     --seed=$PreTrainSeed \
#     --lr=$LearningRate \
#     --mask_rate=$MaskRate \
#     --output_model_dir="./pretrain_models_hyperopt/";

mkdir -p log/${PreTrainer}_ZINC/

export FineTuneData_List=(bbbp tox21 toxcast sider clintox muv hiv bace)
export ProbeTask=downstream

for FineTuneData in "${FineTuneData_List[@]}"; do
# /nethome/hsun349/GraphEval_dev/pretrain_models_hyperopt/AM/zinc_standard_agent/mask_rate-0.15_seed-1_lr-0.001
    export Checkpoint_Folder="${PreTrainer}/${PreTrainData}/mask_rate-${MaskRate}_seed-${PreTrainSeed}_lr-${LearningRate}"
    echo $Checkpoint_Folder
    # === Embedding Extractions ===
    python src/run_embedding_extraction.py \
        --pretrainer="$PreTrainer" \
        --dataset="$FineTuneData" \
        --input_model_file="./pretrain_models_hyperopt/"$Checkpoint_Folder"/epoch99_model_complete.pth" \
        --embedding_dir="./embedding_dir_x/$Checkpoint_Folder/";
    # === Probing ===
    python src/run_validation.py \
        --pretrainer="$PreTrainer" \
        --dataset="$FineTuneData" \
        --probe_task="$ProbeTask" \
        --embedding_dir="./embedding_dir_x/${Checkpoint_Folder}/";
done >> log/${PreTrainer}_ZINC/${PreTrainData}_mask_rate-${MaskRate}_seed-${PreTrainSeed}_lr-${LearningRate}.log

export Seed_List=(1 2 3)
for Seed in "${Seed_List[@]}"; do
for FineTuneData in "${FineTuneData_List[@]}"; do
    export Checkpoint_Folder="${PreTrainer}/${PreTrainData}/mask_rate-${MaskRate}_seed-${PreTrainSeed}_lr-${LearningRate}"
    echo $Checkpoint_Folder
    # === Fine-Tuning the Models ===
    python src/run_validation.py \
        --val_task="finetune" \
        --seed="${Seed}" \
        --dataset=$FineTuneData \
        --pretrainer="FineTune" \
        --probe_task=$ProbeTask \
        --input_model_file="./pretrain_models_hyperopt/"$Checkpoint_Folder"/epoch99_model_complete.pth" ;
done >> log/${PreTrainer}_ZINC/${PreTrainData}_mask_rate-${MaskRate}_seed-${Seed}_lr-${LearningRate}_FineTune.log
done